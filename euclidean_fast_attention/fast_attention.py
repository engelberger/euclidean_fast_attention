import e3x
from e3x.nn.modules import initializers

import numpy as np

from flax import linen as nn

import jax
import jax.numpy as jnp
import jaxtyping

from typing import Any, Callable, Optional, Sequence, Union
from . import tensor_integration
from . import rope


InitializerFn = initializers.InitializerFn
Array = jaxtyping.Array
Bool = jaxtyping.Bool
Float = jaxtyping.Float
Integer = jaxtyping.Integer
UInt32 = jaxtyping.UInt32
Shape = Sequence[Union[int, Any]]
Dtype = Any  # This could be a real type if support for that is added.
PRNGKey = UInt32[Array, '2']
PrecisionLike = jax.lax.PrecisionLike


def frequency_init_fn(
        rng, num_frequencies, num_features, max_frequency, max_length, dtype
):
    """Init function for Euclidean Rope frequencies.

    Args:
    rng: jax.PRNGKey
    num_frequencies: Number of frequencies.
    num_features: Number of features.
    max_frequency: Maximal frequency.
    max_length: Maximal length.
    dtype:

    Returns:
    Vector of frequency values from `[0, ..., max_frequency/max_length]`.
    """
    if num_features // 2 > 1:
        return (
                jnp.linspace(0, max_frequency, int(num_features / 2), dtype=dtype)
                / max_length
        )
    else:
        return jnp.array([max_frequency], dtype=dtype) / max_length


class EuclideanFastAttention(nn.Module):
    lebedev_num: int = 6
    parametrized: bool = True

    num_features_qk: Optional[int] = None
    max_degree_qk: Optional[int] = None
    include_pseudotensors_qk: Optional[bool] = None

    num_features_v: Optional[int] = None
    max_degree_v: Optional[int] = None
    include_pseudotensors_v: Optional[bool] = None

    activation_fn: Optional[Callable[..., Any]] = lambda u: u

    tensor_integration: bool = False
    ti_max_degree_sph: Optional[int] = None
    ti_include_pseudotensors: Optional[bool] = None
    ti_max_degree: Optional[int] = None
    ti_parametrize_coupling_paths: bool = False
    ti_degree_scaling_constants: Optional[Sequence[float]] = None

    epe_frequencies_init_fn: Optional[Callable[..., Any]] = frequency_init_fn
    epe_num_frequencies: Optional[int] = None
    epe_max_frequency: Optional[float] = None
    epe_max_length: Optional[float] = None
    epe_frequencies_trainable: bool = False

    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
            self,
            inputs: Array,
            positions: Array,
            batch_segments: Array,
            graph_mask: Array
    ):

        max_degree_inputs = int(np.rint(np.sqrt(inputs.shape[-2]) - 1).item())

        # if no tensor integration is performed, max_degree_sph can not be set.
        if not self.tensor_integration:

            if self.ti_max_degree_sph is not None:
                raise ValueError(
                    'ti_max_degree_sph can only be set if tensor_integration=True. '
                    f'received {self.tensor_integration=} and {self.ti_max_degree_sph=}'
                )

            if self.ti_max_degree is not None:
                raise ValueError(
                    'ti_max_degree can only be set if tensor_integration=True. '
                    f'received {self.tensor_integration=} and {self.ti_max_degree=}'
                )

            if self.ti_include_pseudotensors is not None:
                raise ValueError(
                    'ti_include_pseudotensors can only be set if tensor_integration=True. '
                    f'received {self.tensor_integration=} and {self.ti_include_pseudotensors=}'
                )

            if self.ti_parametrize_coupling_paths:
                raise ValueError(
                    f'one can only parametrize coupling paths if tensor_integration=True. '
                    f'received {self.tensor_integration=} and {self.ti_parametrize_coupling_paths=}'
                )

        num_features_qk = (
            inputs.shape[-1]
            if self.num_features_qk is None
            else self.num_features_qk
        )

        num_features_v = (
            inputs.shape[-1]
            if self.num_features_v is None
            else self.num_features_v
        )

        if self.parametrized:
            q = e3x.nn.Dense(
                num_features_qk,
                dtype=self.param_dtype
            )(
                inputs
            )  # (N, 1 or 2, (max_degree_in + 1)**2, qk_num_features)
            k = e3x.nn.Dense(
                num_features_qk,
                dtype=self.param_dtype
            )(
                inputs
            )  # (N, 1 or 2, (max_degree_in + 1)**2, qk_num_features)
            v = e3x.nn.Dense(
                num_features_v,
                dtype=self.param_dtype
            )(
                inputs
            )  # (N, 1 or 2, (max_degree_in + 1)**2, qk_num_features)
        else:
            if self.num_features_qk is not None:
                raise ValueError(
                    "Down projection of query and key to `qk_num_features ="
                    f" {self.num_features_qk} is only possible with `parametrized ="
                    " True`."
                )
            if self.num_features_v is not None:
                raise ValueError(
                    "Down projection of value to `v_num_features ="
                    f" {self.num_features_v}` is only possible with `parametrized ="
                    " True`."
                )
            q = inputs  # (N, 1 or 2, (max_degree_in + 1)**2, num_features)
            k = inputs  # (N, 1 or 2, (max_degree_in + 1)**2, num_features)
            v = inputs  # (N, 1 or 2, (max_degree_in + 1)**2, num_features)

        # Lebedev grid.
        with jax.ensure_compile_time_eval():
            grid_u, grid_w = e3x.so3.lebedev_quadrature(
                num=self.lebedev_num
            )

        # If frequencies are trainable, initialize them as params.
        if self.epe_frequencies_trainable:
            frequencies = self.param(
                "frequencies",
                self.epe_frequencies_init_fn,
                self.epe_num_frequencies,
                num_features_qk,
                self.epe_max_frequency,
                self.epe_max_length,
                self.param_dtype,
            )
        # Otherwise just call the init function for the frequencies.
        else:
            frequencies = self.epe_frequencies_init_fn(
                None,  # no RNG key needed.
                self.epe_num_frequencies,
                num_features_qk,
                self.epe_max_frequency,
                self.epe_max_length,
                self.param_dtype,
            )

        # Perform the linear scaling attention aggregation.
        beta = rope.apply(
            q=self.activation_fn(q),
            k=self.activation_fn(k),
            v=v,
            pos=positions,
            theta=frequencies,
            grid_u=grid_u,
            grid_w=grid_w,
            batch_segments=batch_segments,
            graph_mask=graph_mask,
            include_pseudotensors_qk=self.include_pseudotensors_qk,
            include_pseudotensors_v=self.include_pseudotensors_v,
            max_degree_qk=self.max_degree_qk,
            max_degree_v=self.max_degree_v,
            # Determines values at grid points are present or already summed over.
            do_integration=not self.tensor_integration,
        )  # (N, M, P, L, F) or (N, P, L, F)

        # If no tensor integration is required, return beta which is already numerically integrated.
        if not self.tensor_integration:
            return beta

        # Perform tensor product integration. It first builds tensor product between beta and
        # the spherical harmonics expansion for all grid points and then sums over the grid points.
        else:
            # if tensor integration is performed, max_degree_sph is either set explicitly or set
            # to maximal input degree.
            if self.ti_max_degree_sph is None:
                ti_max_degree_sph = max_degree_inputs
            else:
                ti_max_degree_sph = self.ti_max_degree_sph

            # output degree is either set explicitly or set to maximal input degree.
            if self.ti_max_degree is None:
                ti_max_degree = max_degree_inputs
            else:
                ti_max_degree = self.ti_max_degree

            # include_pseudotensors
            if self.ti_include_pseudotensors is None:
                ti_include_pseudotensors = inputs.shape[-3] == 2
            else:
                ti_include_pseudotensors = self.ti_include_pseudotensors

            # build imag_unit * q by permutation and sign flip for imag part in query.
            # imag_q = q#jnp.stack([-q[..., 1::2], q[..., ::2]], axis=-1).reshape(q.shape)
            # imag_beta = rope.apply(
            #     q=self.activation_fn(imag_q),
            #     k=self.activation_fn(k),
            #     v=v,
            #     pos=positions,
            #     theta=frequencies,
            #     grid_u=grid_u,
            #     grid_w=grid_w,
            #     batch_segments=batch_segments,
            #     graph_mask=graph_mask,
            #     include_pseudotensors_qk=self.include_pseudotensors_qk,
            #     include_pseudotensors_v=self.include_pseudotensors_v,
            #     max_degree_qk=self.max_degree_qk,
            #     max_degree_v=self.max_degree_v,
            #     # Determines values at grid points are present or already summed over.
            #     do_integration=False,
            # )  # (N, M, P, L, F) or (N, P, L, F)
            #
            # beta = e3x.nn.add(beta, imag_beta)

            # Expand grid points in spherical harmonics basis.
            grid_u_sph = jnp.expand_dims(
                e3x.so3.spherical_harmonics(
                    grid_u,
                    max_degree=ti_max_degree_sph
                ),
                axis=(-3, -1)
            )  # (M, 1, (L_Y+1)**2), 1)

            if self.ti_degree_scaling_constants is not None:
                if len(self.ti_degree_scaling_constants) != ti_max_degree_sph + 1:
                    raise ValueError(
                        f'the number of constants in `ti_degree_scaling_constants` must equal the number '
                        f'of degrees in the spherical harmonics vector. '
                        f'received {self.ti_degree_scaling_constants=} and {ti_max_degree_sph=}.'
                    )

                repeats = np.array([2*o+1 for o in range(ti_max_degree_sph + 1)])
                c = jnp.array(self.ti_degree_scaling_constants)  # (L_Y+1, )
                c = jnp.repeat(c, repeats=repeats, total_repeat_length=repeats.sum())  # ((L_Y+1)**2, )

                grid_u_sph = grid_u_sph * c[None, None, :, None]

            # Calculate the tensor product between beta and the spherical harmonics expansion on the grid
            # and then integrate.
            return tensor_integration.TensorIntegration(
                include_pseudotensors=ti_include_pseudotensors,
                max_degree=ti_max_degree,
                parametrize_coupling_paths=self.ti_parametrize_coupling_paths
            )(
                jnp.repeat(grid_u_sph, axis=-1, repeats=beta.shape[-1]),
                beta,
                grid_w
            )  # (N, M, P, (L_out+1)**2, F)
