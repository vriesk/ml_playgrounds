#!/usr/bin/python

import argparse
import collections
import numpy as np


def __null_generator(row_count, row_size):
    return np.zeros((row_count, row_size))


class StandardNoiser(object):
    def __init__(self, rng, min_stddev, max_stddev):
        self.__rng = rng
        self.__min_stddev = min_stddev
        self.__max_stddev = max_stddev

    def __call__(self, row_count, row_size):
        out = np.zeros((row_count, row_size))
        for i in range(row_count):
            stddev = self.__rng.uniform(low=self.__min_stddev, high=self.__max_stddev)
            out[i] = self.__rng.normal(scale=stddev, size=row_size)
        return out


Period = collections.namedtuple('Period', ('value', 'scaling'))


class StandardPeriodSampler(object):
    def __init__(self, rng, min_period, max_period, scaling_stddev):
        self.__rng = rng
        self.__min_period = min_period
        self.__max_period = max_period
        self.__scaling_stddev = scaling_stddev

    def __call__(self):
        period = self.__rng.uniform(low=self.__min_period, high=self.__max_period)
        scaling = self.__rng.lognormal(sigma=self.__scaling_stddev)
        return Period(value=period, scaling=scaling)


def generate_wave_sample(rng, period, size):
    # Build the returned array, initially populating it with just the index value for each
    # point. We then transform the values to have the meaning of the value in each of
    # of the points representing the signal iteration at this timepoint.
    sample = np.arange(0, size, dtype=np.float64)
    phase = rng.random() * period.value
    sample += phase

    # Period scaling is non-trivial here.
    # Scaling with a factor <1.0 results in the geometric series with that factor being
    # convergent to a certain number.
    #
    # What it means for the signal shape is that the period reaches 0 (and frequency becomes
    # infinite) at some timepoint T_inf after the 0-time (first sample). Which is fine for signal
    # purposes, as we can just assume that after T the signal just disappears, leaving
    # only noise.
    #
    # However, we're calculating signal values for given timepoints. Therefore, for t < T_inf,
    # we need a formula to calculate a reverse of geometric series sum in order to
    # be able to get the accumulated signal cycles (which will be passed to sin() to get the value).
    # In addition, the values of this reverse sum are not integers, but the formula must be
    # extended to real numbers. Fortunately, the we have a way to do so.
    #
    # The geometric series sum formula is: S_n = a (1 - r^n) / (1-r)   (converges to a/(1-r) for infinity)
    # The reversed (solved for n) formula is therefore: n = log_r (1 - S_n * (1-r) / a)
    # Here:
    #   - r is our scaling factor
    #   - S_n is our timepoint index
    #   - a is our period expressed as timepoint index multiplicator
    # The additional good part is that the formula works equally well for negative numbers, so instead of
    # writing two codepaths, we can simply negate the argument and reverse the scaling factor.
    if period.scaling != 1.0:
      # Let's not deal with non-converging series (infinite sums).
      if period.scaling > 1.0:
          period.scaling = 1.0/scaling
          np.negative(out, out=out)

      signal_cutoff = period.value / (1 - period.scaling)

      np.piecewise(sample,
              [sample >= signal_cutoff],
              [
                  lambda x: 0., # No signal after reaching the limit.
                  lambda x: np.log(1 - x * (1 - period.scaling) / period.value) / np.log(period.scaling),
              ],
              out=sample)
    else:
        sample /= period.value

    return sample


# Returns sample_count x (signal_size + prediction_size) NumPy array.
#
# Period corresponds to the initial frequency of the sampled data.
# It's expressed in sampling rate units (number of sampling points
# that constitute the full signal period), can be a float value.
#
# Period scaling factor indicates the exponential period scaling of each
# susbsequent signal repetition after the initial one.
#
# The jitter is understood to affect each point by altering the actual
# point's generation timestampy by the jitter amount, with value of 1.0 meaning
# the timepoint is fully shifted to the next sampling point (and -1.0 to the
# previous one accordingly).
def generate_wave_samples(
        rng, sample_count, signal_size, prediction_size,
        period_sampler=lambda: Period(value=1.0, scaling=1.0),
        noise_generator=__null_generator, jitter_generator=__null_generator):

    row_size = signal_size + prediction_size
    out = np.zeros((sample_count, row_size))

    for i in range(sample_count):
        out[i] = generate_wave_sample(rng=rng, period=period_sampler(), size=row_size)
    
    # The prediction segment of the sample should not have any noise and jitter, therefore we only
    # ask the generators to generate noise/jitter for the non-prediction segment. For the algebra
    # to work, we need to pad those before adding.
    def resize_with_prediction(a):
        return np.lib.pad(a, ((0,0), (0, prediction_size)), 'constant', constant_values=(0.))

    # Apply jitter. As jitter is not subject to scaling, it must be applied after scaling itself.
    jitter = jitter_generator(row_count=sample_count, row_size=signal_size)
    np.add(out, resize_with_prediction(jitter), out=out)

    np.sin(2 * np.pi * out, out=out)

    # Finally, apply noise.
    noise = noise_generator(row_count=sample_count, row_size=signal_size)
    np.add(out, resize_with_prediction(noise), out=out)

    return out


def __get_args():
    parser = argparse.ArgumentParser("Wavey Data Generator")
    parser.add_argument('--sample_count', type=int, default=None, required=True, help="Number of samples to produce")
    parser.add_argument('--signal_size', type=int, default=None, required=True, help="Size of the signal segment of the sample.")
    parser.add_argument('--prediction_size', type=int, default=None, required=True, help="Size of the prediction segment of the sample.")
    parser.add_argument('--rng_seed', type=int, default=None, help="Seed for the RNG. Defaults to system entropy.")
    parser.add_argument('--output', type=str, default=None, required=True, help="Output filename (NPY format).")
    parser.add_argument('--min_period', type=float, default=10.0, help="Lower bound of the period randomization uniform distribution.")
    parser.add_argument('--max_period', type=float, default=1000.0, help="Upper bound of the period randomization uniform distribution.")
    parser.add_argument('--period_scaling_stddev', type=float, default=0., help="Standard deviation for the period scaling lognormal distribution (0 disables scaling)")
    parser.add_argument('--min_noise_stddev', type=float, default=None, help="Lower bound of the noise standard deviation uniform distribution. If omitted, no noise is applied. Must be used together with --max_noise_stddev.")
    parser.add_argument('--max_noise_stddev', type=float, default=None, help="Upper bound of the noise standard deviation uniform distribution. If omitted, no noise is applied. Must be used together with --min_noise_stddev.")
    parser.add_argument('--min_jitter_stddev', type=float, default=None, help="Lower bound of the jitter standard deviation uniform distribution. If omitted, no jitter is applied. Must be used together with --max_jitter_stddev.")
    parser.add_argument('--max_jitter_stddev', type=float, default=None, help="Upper bound of the jitter standard deviation uniform distribution. If omitted, no jitter is applied. Must be used together with --min_jitter_stddev.")

    return parser.parse_args()


if __name__ == "__main__":
    args = __get_args()
    rng = np.random.default_rng(args.rng_seed)

    period_sampler = StandardPeriodSampler(rng=rng, min_period=args.min_period, max_period=args.max_period, scaling_stddev=args.period_scaling_stddev)
    noise_generator = __null_generator
    if args.min_noise_stddev is not None and args.max_noise_stddev is not None:
        noise_generator = StandardNoiser(rng=rng, min_stddev=args.min_noise_stddev, max_stddev=args.max_noise_stddev)
    jitter_generator = __null_generator
    if args.min_jitter_stddev is not None and args.max_jitter_stddev is not None:
        jitter_generator = StandardNoiser(rng=rng, min_stddev=args.min_jitter_stddev, max_stddev=args.max_jitter_stddev)

    data = generate_wave_samples(
            rng=rng,
            sample_count=args.sample_count,
            signal_size=args.signal_size,
            prediction_size=args.prediction_size,
            period_sampler=period_sampler,
            noise_generator=noise_generator,
            jitter_generator=jitter_generator)

    np.save(args.output, data, allow_pickle=False)

