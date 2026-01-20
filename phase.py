"""
Endpoint Corrected Hilbert Transform (ECHT) with optional calibration.
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
from scipy.signal import butter, bessel, cheby1, cheby2, ellip, freqz
from scipy.fft import fft, ifft, fftshift, ifftshift, next_fast_len


class ECHT:
    """
    Endpoint-Corrected Hilbert Transform (ECHT) with optional endpoint calibration.

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

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
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
    ):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.sfreq = sfreq
        self.n_fft = n_fft
        self.fft_mode = str(fft_mode).lower()
        if self.fft_mode not in ("fast", "exact"):
            raise ValueError("fft_mode must be 'fast' or 'exact'")
        self.filt_order = filt_order
        self.filter_type = filter_type

        self.calibrate = bool(calibrate)
        self.f0 = f0

        # Runtime attributes set during fitting
        self.h_ = None          # analytic-signal multiplier
        self.coef_ = None       # band-pass frequency response (column vector)
        self.calib_gain_ = None # complex scalar calibration factor
        self.calib_err_ = None  # dict with theoretical error metrics


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _design_bandpass(l_freq, h_freq, sfreq, filt_order, n_fft, filter_type="butter"):
        """
        Construct Hilbert multiplier and centered band-pass frequency response.

        Returns
        -------
        h : ndarray
            Analytic-signal Hilbert multiplier (length ``n_fft``).
        H_center : ndarray
            Band-pass filter response on a centered FFT grid.
        """

        # ---------------------------
        # Hilbert analytic multiplier
        # ---------------------------
        h = np.zeros(n_fft, dtype=float)
        h[0] = 1
        if n_fft % 2 == 0:
            # even length: DC + positive + Nyquist
            h[1:n_fft // 2] = 2
            h[n_fft // 2] = 1
        else:
            # odd length: DC + positive
            h[1:(n_fft + 1) // 2] = 2

        # ---------------------------
        # Band-pass IIR filter design
        # ---------------------------
        Wn = [l_freq / (sfreq / 2), h_freq / (sfreq / 2)]

        if filter_type == "butter":
            b, a = butter(filt_order, Wn, btype="band")
        elif filter_type == "bessel":
            b, a = bessel(filt_order, Wn, btype="band", norm="phase")
        elif filter_type == "cheby1":
            b, a = cheby1(filt_order, 1, Wn, btype="band")  # 1 dB ripple
        elif filter_type == "cheby2":
            b, a = cheby2(filt_order, 40, Wn, btype="band")  # 40 dB stopband
        elif filter_type == "ellip":
            b, a = ellip(filt_order, 1, 40, Wn, btype="band")  # elliptic / Cauer
        else:
            raise ValueError(f"Unknown filter_type: {filter_type}")

        # Centered frequency grid (Hz), length n_fft
        filt_freq = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1 / sfreq))
        _, H_center = freqz(b, a, worN=filt_freq, fs=sfreq)

        return h, H_center

    @staticmethod
    def _calibration(f0, sfreq, N, l_freq, h_freq, filt_order=1, filter_type="butter", L=None, fft_mode="fast"):
        """
        Compute theoretical endpoint error for a pure cosine input.
        For many applications this approximation is valid.

        Returns
        -------
        err : dict
            Contains:
            - 'Gplus'  : desired analytic-aligned component
            - 'Gminus' : residual image component
            - 'C_opt'  : complex calibration gain
            - 'J_opt'  : minimal mean-square endpoint error
        """
        if L is None:
            L = next_fast_len(N) if fft_mode == "fast" else N

        # ---- Dirichlet kernel (length-N cosine spectrum) ----
        def _dirichlet_N(alpha):
            alpha = np.asarray(alpha, dtype=float)
            D = np.empty(alpha.shape, dtype=np.complex128)
            small = np.abs(alpha) < 1e-12
            D[small] = N
            a_ns = alpha[~small]
            D[~small] = (
                np.exp(1j * a_ns * (N - 1) / 2)
                * np.sin(0.5 * N * a_ns)
                / np.sin(0.5 * a_ns)
            )
            return D

        k = np.arange(L)
        omega_k = 2 * np.pi * k / L  # rad/sample, DFT grid
        omega0 = 2 * np.pi * f0 / sfreq
        n = N - 1

        # Hilbert multiplier + centered band-pass response
        h, H_center = ECHT._design_bandpass(
            l_freq=l_freq, h_freq=h_freq, sfreq=sfreq, filt_order=filt_order, n_fft=L, filter_type=filter_type,
        )

        H_eff = ifftshift(H_center)
        G = h * H_eff

        D_plus = _dirichlet_N(omega0 - omega_k )
        D_minus = _dirichlet_N(-omega0 - omega_k)

        X_plus = 0.5 * D_plus
        X_minus = 0.5 * D_minus

        # Endpoint of the ECHT output for each spectral component.
        phase = np.exp(1j * omega_k * n)
        P = (G * X_plus * phase).sum() / L
        M = (G * X_minus * phase).sum() / L

        # Separate contributions aligned with the ideal analytic signal.
        Gplus = P * np.exp(-1j * omega0 * n)
        Gminus = M * np.exp(-1j * omega0 * n)

        # MSE-optimal complex calibration gain and corresponding minimal error
        denom = np.abs(Gplus) ** 2 + np.abs(Gminus) ** 2
        if denom == 0:
            C_opt = 1 + 0j
            J_opt = 0
        else:
            C_opt = np.conj(Gplus) / denom
            J_opt = np.abs(Gminus) ** 2 / denom

        return {
            "Gplus": Gplus,
            "Gminus": Gminus,
            "C_opt": C_opt,
            "J_opt": J_opt,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        """
        Fit the ECHT transform to the input signal (FFT size, filters, calibration).

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_channels)
            The input signal to be transformed.

        Returns
        -------
        self
        """
        X = np.asarray(X)
        n_samples = X.shape[0]

        if self.n_fft is None:
            self.n_fft = (
                next_fast_len(n_samples) if self.fft_mode == "fast" else n_samples
            )

        self.h_, H_center = self._design_bandpass(
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            sfreq=self.sfreq,
            filt_order=self.filt_order,
            n_fft=self.n_fft,
            filter_type=self.filter_type,
        )
        self.coef_ = H_center[:, None]

        # Optional endpoint calibration
        self.calib_gain_ = None
        self.calib_err_ = None
        if self.calibrate:
            if self.f0 is None:
                raise ValueError("f0 (signal frequency) must be provided when calibrate=True.")

            err = self._calibration(
                f0=self.f0,
                sfreq=self.sfreq,
                N=n_samples,
                l_freq=self.l_freq,
                h_freq=self.h_freq,
                filt_order=self.filt_order,
                L=self.n_fft,
                fft_mode=self.fft_mode,
                filter_type=self.filter_type,
            )
            C_opt = err["C_opt"]

            if not (np.isfinite(C_opt.real) and np.isfinite(C_opt.imag)):
                raise RuntimeError(
                    "Non-finite calibration gain C_opt computed for ECHT."
                )

            self.calib_gain_ = C_opt
            self.calib_err_ = err

        return self

    def transform(self, X):
        """Apply the ECHT transform to the input signal.

        Parameters
        ----------
        X : ndarray, shape=(n_samples, n_channels)
            The input signal to be transformed.

        Returns
        -------
        Xf : ndarray, shape=(n_samples, n_channels)
            The transformed signal (complex-valued).

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

        # ---------------------------
        # Frequency-domain
        # ---------------------------
        Xf = fft(X, self.n_fft, axis=0)

        # In contrast to :meth:`scipy.signal.hilbert()`, the code then
        # multiplies the array by a frequency response vector of a causal
        # bandpass filter.
        Xf = Xf * self.h_[:, None]

        # The array is arranged, using fft_shift function, so that the zero-frequency
        # component is at the center of the array, before the multiplication, and
        # rearranged back so that the zero-frequency component is at the left of the
        # array using ifft_shift(). Finally, the IFFT is computed.
        Xf = fftshift(Xf, axes=0)
        Xf = Xf * self.coef_
        Xf = ifftshift(Xf, axes=0)
        Xf = ifft(Xf, axis=0)

        # Optional global endpoint calibration
        if self.calibrate and self.calib_gain_ is not None:
            Xf = Xf * self.calib_gain_

        # Truncate to original number of samples
        Xf = Xf[:n_samples, :]

        return Xf

    def fit_transform(self, X, y=None):
        """Fit the ECHT transform to the input signal and transform it."""
        return self.fit(X).transform(X)
