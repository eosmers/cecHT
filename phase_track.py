"""
Endpoint Corrected Hilbert Transform (ECHT) with optional calibration and frequency tracking.
===========================================

Implements a causal, band-limited, endpoint-corrected analytic transform.
This method can be used for example to estimate the real-time phase of a signal.

This code is based on MEEGkit's ECHT class
    https://nbara.github.io/python-meegkit/
    https://github.com/nbara/python-meegkit/tree/master
    BSD 3-Clause License
    Copyright (c) 2019, nbara

ECHT was originally invented by Schreglmann et al. (2021):
Schreglmann, S. R., Wang, D., Peach, R. L., Li, J., Zhang, X.,
       Latorre, A., ... & Grossman, N. (2021). Non-invasive suppression of
       essential tremor via phase-locked disruption of its temporal coherence.
       Nature communications, 12(1), 363.
"""

import numpy as np
from scipy.fft import fft, ifft, fftshift, ifftshift, next_fast_len

from phase import ECHT as _ECHT


class ECHT(_ECHT):
    """
    Endpoint-Corrected Hilbert Transform (ECHT) with optional endpoint calibration.

    This module reuses the base implementation from ``phase.py`` and adds:
    - Optional dynamic band-pass tracking around the current center frequency ``f0``.
    - Optional per-window endpoint calibration gain lookup (cached).

    Parameters
    ----------
    l_freq : float or None
        Low cutoff of band-pass filter (Hz). If None, only high-pass is applied.
    h_freq : float or None
        High cutoff of band-pass filter (Hz). If None, only low-pass is applied.
    sfreq : float
        Sampling rate of the signal.
    n_fft : int or None, optional
        FFT length used internally. If None, determined by ``fft_mode``.
    fft_mode : {'fast', 'exact'}, optional
        Strategy for FFT length selection:
        - 'fast' (standard): use ``next_fast_len`` (faster, potentially less accurate)
        - 'exact': use ``n_samples`` (slower, exact window alignment)
    filt_order : int, optional
        IIR filter order (default 1).
        Note: scipy defines a bandpass of order 1 as (highpass + lowpass both order 1).
    filter_type : {'butter','bessel','cheby1','cheby2','ellip'}, optional
        Type of band-pass filter.
        butter: standard
    calibrate : bool, optional
        If True, apply complex-valued scalar gain to correct endpoint amplitude/phase.
    f0 : float or None, optional
        Center frequency (Hz) of the assumed cosine used for calibration.

    bandpass_tracking : bool, optional
        If True, update the band-pass edges around the current f0 (default False).
    bandpass_update_mode : {'threshold', 'always'}, optional
        - 'threshold': update only when |f0 - last_f0| >= bandpass_f0_thresh_hz
        - 'always': update band-pass for every call when f0 is provided
    bandpass_f0_thresh_hz : float, optional
        Threshold (Hz) used when ``bandpass_update_mode='threshold'``.
    bandpass_bw_hz : float or None, optional
        Fixed band-pass bandwidth (Hz) used for tracking. If None, uses initial
        (h_freq - l_freq) determined at fit-time.
    bandpass_min_hz : float, optional
        Clamp lower edge of the tracked band-pass (Hz).
    bandpass_max_hz : float or None, optional
        Clamp upper edge of the tracked band-pass (Hz). If None, uses (Nyquist - 0.1).

    Attributes
    ----------
    h_ : ndarray
        Hilbert analytic-signal multiplier (freq-domain).
    coef_ : ndarray
        Frequency response of the band-pass filter (column vector).
    calib_gain_ : complex or None
        Global endpoint calibration gain.
    calib_err_ : dict or None
        Theoretical calibration error metrics.
    """

    def __init__(
            self,
            l_freq,
            h_freq,
            sfreq,
            n_fft=None,
            fft_mode="fast",
            filt_order=1,
            filter_type="butter",
            calibrate=False,
            f0=None,
    # --- optional dynamic bandpass tracking ---
            bandpass_tracking=False,
            bandpass_update_mode="threshold",  # "threshold" or "always"
            bandpass_f0_thresh_hz=1,  # used if mode="threshold"
            bandpass_bw_hz=None,  # if None, uses initial (h_freq-l_freq)
            bandpass_min_hz=0.1,  # clamp low edge
            bandpass_max_hz=None,  # clamp high edge (None -> Nyquist-0.1)
    ):
        # Reuse ECHT; keep behavior aligned with the original phase_track:
        # - FFT length selection uses "fast" behavior (next_fast_len in base)
        super().__init__(
            l_freq=l_freq,
            h_freq=h_freq,
            sfreq=sfreq,
            n_fft=n_fft,
            fft_mode="fast",
            filt_order=filt_order,
            filter_type=filter_type,
            calibrate=calibrate,
            f0=f0,
        )

        # Cache calibration gains for dynamic f0 tracking
        self._calib_cache_ = {}

        # --- bandpass tracking config/state ---
        self.bandpass_tracking = bool(bandpass_tracking)
        self.bandpass_update_mode = str(bandpass_update_mode).lower()
        if self.bandpass_update_mode not in ("threshold", "always"):
            raise ValueError("bandpass_update_mode must be 'threshold' or 'always'.")

        self.bandpass_f0_thresh_hz = float(bandpass_f0_thresh_hz)
        self.bandpass_bw_hz = None if bandpass_bw_hz is None else float(bandpass_bw_hz)
        self.bandpass_min_hz = float(bandpass_min_hz)
        self.bandpass_max_hz = None if bandpass_max_hz is None else float(bandpass_max_hz)

        # last f0 at which we UPDATED the bandpass (hysteresis reference)
        self._bp_last_f0_ = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _update_bandpass(self, l_freq: float, h_freq: float):
        """
        Update band-pass dependent state without re-fitting.

        This updates:
        - ``l_freq`` / ``h_freq``
        - ``coef_`` (centered band-pass response used after fftshift)
        - invalidates calibration state that depends on the previous band-pass
        """
        if self.n_fft is None:
            raise RuntimeError("Call fit() once before updating the bandpass (n_fft must be known).")

        l_freq = float(l_freq)
        h_freq = float(h_freq)

        if not (np.isfinite(l_freq) and np.isfinite(h_freq)):
            raise ValueError("Non-finite l_freq/h_freq in _update_bandpass().")
        if l_freq <= 0 or h_freq <= 0 or h_freq <= l_freq:
            raise ValueError(f"Invalid bandpass: l_freq={l_freq}, h_freq={h_freq}")

        # If nothing changes, do nothing
        if (self.l_freq is not None) and (self.h_freq is not None):
            if np.isclose(l_freq, float(self.l_freq)) and np.isclose(h_freq, float(self.h_freq)):
                return

        self.l_freq = l_freq
        self.h_freq = h_freq

        # Keep BW updated (when tracking uses a fixed bandwidth)
        if self.bandpass_bw_hz is not None:
            self.bandpass_bw_hz = float(self.h_freq - self.l_freq)

        # Recompute frequency-domain responses for the NEW bandpass
        # (h_ is deterministic given n_fft, but recomputing it is cheap and safe)
        self.h_, H_center = self._design_bandpass(
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            sfreq=self.sfreq,
            filt_order=self.filt_order,
            n_fft=self.n_fft,
            filter_type=self.filter_type,
        )
        self.coef_ = H_center[:, None]

        # Anything depending on the previous band-pass is now invalid
        self._calib_cache_.clear()
        self.calib_gain_ = None
        self.calib_err_ = None

    def _update_bandpass_for_f0(self, f0_used: float):
        """Update the band-pass around f0 if tracking is enabled."""
        if not self.bandpass_tracking:
            return

        f0_used = float(f0_used)
        if not np.isfinite(f0_used) or f0_used <= 0:
            return

        # Decide BW (fixed) and clamp edges
        bw = self.bandpass_bw_hz
        if bw is None:
            bw = float(self.h_freq - self.l_freq)

        nyq = 0.5 * float(self.sfreq)
        fmin = float(self.bandpass_min_hz)
        fmax = self.bandpass_max_hz
        fmax = float(fmax) if fmax is not None else (nyq - 0.1)

        l_new = max(fmin, f0_used - 0.5 * bw)
        h_new = min(fmax, f0_used + 0.5 * bw)
        if h_new <= l_new:
            return

        if self.bandpass_update_mode == "always":
            do_update = True
        else:
            thresh = float(self.bandpass_f0_thresh_hz)
            last = self._bp_last_f0_
            if last is None or not np.isfinite(last):
                last = f0_used
                self._bp_last_f0_ = float(last)
            do_update = (abs(f0_used - float(last)) >= thresh)

        if do_update:
            self._update_bandpass(l_new, h_new)
            self._bp_last_f0_ = float(f0_used)

    def _calib_gain_lut(self, f0: float, N: int):
        """
        Return (gain, err) for the current configuration.

        Uses a small cache keyed by (f0, N, bandpass/filter params, sfreq, n_fft).
        """
        if f0 is None:
            return self.calib_gain_, self.calib_err_

        f0_key = float(np.round(float(f0), 6))

        # Include anything that changes the theoretical G+/G- and thus C_opt
        key = (
            f0_key,
            int(N),
            float(np.round(float(self.l_freq), 6)),
            float(np.round(float(self.h_freq), 6)),
            int(self.filt_order),
            int(self.n_fft),
            str(self.filter_type),
            float(self.sfreq),
        )

        hit = self._calib_cache_.get(key, None)
        if hit is not None:
            return hit

        err = self._calibration(
            f0=f0_key,
            sfreq=self.sfreq,
            N=int(N),
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            filt_order=self.filt_order,
            L=self.n_fft,
            filter_type=self.filter_type,
        )
        C_opt = err["C_opt"]
        self._calib_cache_[key] = (C_opt, err)
        return C_opt, err

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        """
        Fit the ECHT transform to the input signal.

        This calls phase.ECHT.fit and initializes tracking-related
        state (bandwidth inference, caches).
        """
        X = np.asarray(X)
        self._fit_N_ = int(X.shape[0])

        # If tracking needs a bandwidth and none was specified, infer it once
        if self.bandpass_bw_hz is None and self.l_freq is not None and self.h_freq is not None:
            self.bandpass_bw_hz = float(self.h_freq - self.l_freq)

        # Base fit computes: n_fft, h_, coef_, and optional fit-time calib
        out = super().fit(X, y=y)

        # Tracking-specific caches
        self._calib_cache_.clear()
        self._bp_last_f0_ = None

        return out

    def transform(self, X, f0=None):
        """Apply the ECHT transform to the input signal.

        Parameters
        ----------
        X : ndarray, shape=(n_samples, n_channels)
        f0 : float | None
            If provided and calibrate=True, compute and apply the complex scalar
            calibration gain for this f0 (tracked per window).

        Returns
        -------
        Xf : ndarray, shape=(n_samples, n_channels)
        """
        X = np.asarray(X)

        if not np.isrealobj(X):
            X = np.real(X)

        # if not fitted
        if self.h_ is None or self.coef_ is None:
            self.fit(X)

        if X.ndim == 1:
            X = X[:, np.newaxis]

        n_samples = X.shape[0]
        if n_samples > self.n_fft:
            raise ValueError(f"Input length {n_samples} exceeds n_fft {self.n_fft}.")

        # --- optional dynamic bandpass tracking ---
        f0_used = f0 if f0 is not None else self.f0
        if f0_used is not None:
            self._update_bandpass_for_f0(float(f0_used))

        # FFT -> analytic multiplier -> centred bandpass -> IFFT
        Xf = fft(X, self.n_fft, axis=0)
        Xf = Xf * self.h_[:, None]
        Xf = fftshift(Xf, axes=0)
        Xf = Xf * self.coef_
        Xf = ifftshift(Xf, axes=0)
        Xf = ifft(Xf, axis=0)

        # Dynamic complex-scalar endpoint calibration
        if self.calibrate:
            gain, _ = self._calib_gain_lut(f0 if f0 is not None else self.f0, n_samples)
            if gain is not None:
                Xf = Xf * gain

        # Truncate
        Xf = Xf[:n_samples, :]
        return Xf

    def fit_transform(self, X, y=None):
        """Fit the ECHT transform to the input signal and transform it."""
        return self.fit(X).transform(X)
