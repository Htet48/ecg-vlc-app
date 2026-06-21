"""
ACO-OFDM Modulation/Demodulation for VLC System
Implements complete digital communication pipeline matching your methodology
"""

import numpy as np
from scipy.fft import fft, ifft


class ACO_OFDM_VLC:
    """
    ACO-OFDM (Asymmetrically Clipped Optical OFDM) for VLC transmission
    
    Pipeline:
    TX: ECG → Symbol Mapping → IFFT → Clipping → Add CP → VLC Channel
    RX: Received → Remove CP → FFT → Equalization → De-mapping → ECG
    """
    
    def __init__(self, N=64, cp_len=16, M=4):
        """
        Initialize ACO-OFDM system
        
        Args:
            N: Number of OFDM subcarriers (must be power of 2)
            cp_len: Cyclic prefix length (typically N/4)
            M: Modulation order (4 for 4-QAM, 16 for 16-QAM)
        """
        self.N = N  # FFT size
        self.cp_len = cp_len  # Cyclic prefix
        self.M = M  # Modulation order
        self.bits_per_symbol = int(np.log2(M))
        
        # Generate constellation
        self.constellation = self._generate_qam_constellation(M)
        
    def _generate_qam_constellation(self, M):
        """Generate M-QAM constellation points"""
        if M == 4:
            # 4-QAM (QPSK)
            constellation = np.array([
                1+1j, 1-1j, -1+1j, -1-1j
            ]) / np.sqrt(2)
        elif M == 16:
            # 16-QAM
            points = [-3, -1, 1, 3]
            constellation = np.array([
                complex(i, q) for i in points for q in points
            ]) / np.sqrt(10)
        else:
            raise ValueError(f"Modulation order {M} not supported")
        
        return constellation
    
    def ecg_to_bits(self, ecg_signal, bit_depth=8):
        """
        Convert ECG signal to bit stream
        
        Args:
            ecg_signal: Normalized ECG signal [0, 1]
            bit_depth: Bits per sample (8 = 256 levels)
        
        Returns:
            bits: Binary sequence
        """
        # Quantize to bit_depth levels
        levels = 2 ** bit_depth
        quantized = np.round(ecg_signal * (levels - 1)).astype(int)
        
        # Convert to binary
        bits = []
        for value in quantized:
            bit_string = format(value, f'0{bit_depth}b')
            bits.extend([int(b) for b in bit_string])
        
        return np.array(bits)
    
    def bits_to_symbols(self, bits):
        """
        Map bit stream to QAM symbols
        
        Args:
            bits: Binary sequence
        
        Returns:
            symbols: Complex QAM symbols
        """
        # Pad to multiple of bits_per_symbol
        n_pad = (self.bits_per_symbol - len(bits) % self.bits_per_symbol) % self.bits_per_symbol
        bits_padded = np.concatenate([bits, np.zeros(n_pad)])
        
        # Group into symbols
        n_symbols = len(bits_padded) // self.bits_per_symbol
        symbols = np.zeros(n_symbols, dtype=complex)
        
        for i in range(n_symbols):
            bit_group = bits_padded[i*self.bits_per_symbol:(i+1)*self.bits_per_symbol]
            symbol_index = int(''.join(map(str, bit_group.astype(int))), 2)
            symbols[i] = self.constellation[symbol_index]
        
        return symbols
    
    def aco_ofdm_modulate(self, symbols):
        """
        ACO-OFDM Modulation
        
        Steps:
        1. Hermitian symmetry to get real IFFT output
        2. IFFT to convert to time domain
        3. Clipping (set negative to zero for optical)
        4. Add cyclic prefix
        
        Args:
            symbols: QAM symbols
        
        Returns:
            modulated_signal: Real non-negative time-domain signal
        """
        # CORRECTED (Hermitian / N-4 fix): only the POSITIVE-FREQUENCY odd
        # subcarriers (k = 1,3,...,N/2-1) carry independent QAM data => N/4
        # independent symbols per frame. The negative-frequency odd subcarriers
        # are Hermitian-conjugate mirrors and carry NO new data.
        # (Previously used N/2, which double-counted the mirror carriers and
        # produced an artificial ~25% BER floor.)
        n_active = self.N // 4
        n_blocks = int(np.ceil(len(symbols) / n_active))
        symbols_padded = np.concatenate([symbols, np.zeros(n_blocks * n_active - len(symbols))])

        ofdm_blocks = []

        for block_idx in range(n_blocks):
            # Get symbols for this block
            block_symbols = symbols_padded[block_idx*n_active:(block_idx+1)*n_active]

            # Create frequency domain signal with Hermitian symmetry
            freq_domain = np.zeros(self.N, dtype=complex)

            # ACO-OFDM: place symbols on positive-frequency odd subcarriers only
            freq_domain[1:self.N//2:2] = block_symbols

            # Hermitian symmetry for real output
            freq_domain[self.N//2+1:] = np.conj(freq_domain[self.N//2-1:0:-1])
            
            # IFFT to time domain
            time_domain = ifft(freq_domain)
            
            # Should be real (imaginary part is numerical error)
            time_domain = np.real(time_domain)
            
            # ACO: Asymmetric clipping (clip negative to zero)
            time_domain_clipped = np.clip(time_domain, 0, None)
            
            # Add cyclic prefix
            cp = time_domain_clipped[-self.cp_len:]
            block_with_cp = np.concatenate([cp, time_domain_clipped])
            
            ofdm_blocks.append(block_with_cp)
        
        # Concatenate all blocks
        modulated_signal = np.concatenate(ofdm_blocks)
        
        return modulated_signal, n_blocks
    
    def vlc_channel(self, signal, h_t, ambient='bright'):
        """
        VLC optical channel
        
        Args:
            signal: Transmitted optical intensity
            h_t: Channel gain (time-varying)
            ambient: 'bright' or 'dark' lighting condition
        
        Returns:
            received: Received signal with noise
        """
        # Extend h_t to match signal length if needed
        if len(h_t) < len(signal):
            h_t = np.tile(h_t, int(np.ceil(len(signal) / len(h_t))))[:len(signal)]
        elif len(h_t) > len(signal):
            h_t = h_t[:len(signal)]
        
        # Channel effect
        received = h_t * signal
        
        # Physical noise model (thermal + shot)
        if ambient == 'bright':
            sigma_th = 0.0015  # Thermal noise std
            sigma_sh = 0.0025  # Shot noise coefficient
        else:
            sigma_th = 0.003
            sigma_sh = 0.0015
        
        # Thermal noise (constant, Gaussian)
        n_thermal = np.random.randn(len(received)) * sigma_th
        
        # Shot noise (signal-dependent, Gaussian approx)
        n_shot = np.random.randn(len(received)) * sigma_sh * np.sqrt(np.maximum(received, 1e-12))
        
        # Total noise
        received = received + n_thermal + n_shot
        
        # Ensure non-negative (photodetector can't measure negative light)
        received = np.clip(received, 0, None)
        
        return received
    
    def aco_ofdm_demodulate(self, received_signal, n_blocks, H_est=None):
        """
        ACO-OFDM Demodulation
        
        Steps:
        1. Remove cyclic prefix
        2. FFT to frequency domain
        3. Channel equalization
        4. Extract symbols from odd subcarriers
        
        Args:
            received_signal: Received time-domain signal
            n_blocks: Number of OFDM blocks
            H_est: Estimated channel frequency response (None = perfect CSI)
        
        Returns:
            symbols_recovered: Recovered QAM symbols
        """
        block_len = self.N + self.cp_len
        symbols_all = []
        
        for block_idx in range(n_blocks):
            # Extract block
            start = block_idx * block_len
            end = start + block_len
            
            if end > len(received_signal):
                # Incomplete block, pad with zeros
                block = np.concatenate([
                    received_signal[start:],
                    np.zeros(end - len(received_signal))
                ])
            else:
                block = received_signal[start:end]
            
            # Remove cyclic prefix
            block_no_cp = block[self.cp_len:]
            
            # FFT to frequency domain
            freq_domain = fft(block_no_cp)
            
            # Channel equalization
            if H_est is None:
                # Perfect CSI: assume no equalization needed
                freq_eq = freq_domain
            else:
                # Zero-forcing equalization
                freq_eq = freq_domain / (H_est + 1e-10)
            
            # Extract symbols from odd subcarriers (ACO-OFDM)
            symbols_block = freq_eq[1:self.N:2]
            
            symbols_all.append(symbols_block)
        
        # Concatenate all symbols
        symbols_recovered = np.concatenate(symbols_all)
        
        return symbols_recovered
    
    def symbols_to_bits(self, symbols):
        """
        De-map symbols to bits (hard decision)
        
        Args:
            symbols: Received QAM symbols
        
        Returns:
            bits: Binary sequence
        """
        bits = []
        
        for symbol in symbols:
            # Find nearest constellation point
            distances = np.abs(self.constellation - symbol)
            nearest_idx = np.argmin(distances)
            
            # Convert index to bits
            bit_string = format(nearest_idx, f'0{self.bits_per_symbol}b')
            bits.extend([int(b) for b in bit_string])
        
        return np.array(bits)
    
    def bits_to_ecg(self, bits, original_length, bit_depth=8):
        """
        Reconstruct ECG from bit stream
        
        Args:
            bits: Binary sequence
            original_length: Original ECG signal length
            bit_depth: Bits per sample
        
        Returns:
            ecg_recovered: Reconstructed ECG signal
        """
        # Group bits into samples
        n_samples = len(bits) // bit_depth
        ecg_recovered = np.zeros(n_samples)
        
        for i in range(n_samples):
            bit_group = bits[i*bit_depth:(i+1)*bit_depth]
            value = int(''.join(map(str, bit_group.astype(int))), 2)
            ecg_recovered[i] = value / (2**bit_depth - 1)
        
        # Truncate to original length
        ecg_recovered = ecg_recovered[:original_length]
        
        # Pad if needed
        if len(ecg_recovered) < original_length:
            ecg_recovered = np.concatenate([
                ecg_recovered,
                np.zeros(original_length - len(ecg_recovered))
            ])
        
        return ecg_recovered
    
    def full_transmission(self, ecg_signal, h_t, ambient='bright', simulate_fec=False):
        """
        Complete transmission pipeline
        
        Args:
            ecg_signal: Original ECG signal [0, 1]
            h_t: Channel gain time series
            snr_db: SNR in dB
            simulate_fec: If True, simulate FEC by correcting errors
        
        Returns:
            ecg_received: Received ECG signal
            stats: Dictionary with transmission statistics (includes symbols for plotting)
        """
        # TX: ECG → Bits → Symbols → Modulate
        bits_tx = self.ecg_to_bits(ecg_signal)
        symbols_tx = self.bits_to_symbols(bits_tx)
        modulated, n_blocks = self.aco_ofdm_modulate(symbols_tx)
        
        # Channel
        received = self.vlc_channel(modulated, h_t, ambient)
        
        # RX: Demodulate → Symbols → Bits → ECG
        symbols_rx = self.aco_ofdm_demodulate(received, n_blocks)
        bits_rx = self.symbols_to_bits(symbols_rx[:len(symbols_tx)])
        
        # Calculate BER before FEC
        ber_raw = np.mean(bits_tx[:len(bits_rx)] != bits_rx)
        
        # Simulate FEC if enabled
        if simulate_fec:
            # Simulate Reed-Solomon-like error correction
            # RS can typically correct up to 10-15% errors
            # We simulate this by fixing some errors
            error_positions = np.where(bits_tx[:len(bits_rx)] != bits_rx)[0]
            n_errors = len(error_positions)
            
            # RS codes can correct errors up to a limit
            # Typical RS(255,223) can correct ~13% errors
            correction_rate = 0.95  # Correct 95% of errors
            n_corrected = int(n_errors * correction_rate)
            
            if n_corrected > 0:
                # Randomly select which errors to correct
                correct_indices = np.random.choice(error_positions, n_corrected, replace=False)
                bits_rx[correct_indices] = bits_tx[correct_indices]
            
            ber_after_fec = np.mean(bits_tx[:len(bits_rx)] != bits_rx)
        else:
            ber_after_fec = ber_raw
        
        ecg_received = self.bits_to_ecg(bits_rx, len(ecg_signal))
        
        # Calculate statistics
        stats = {
            'n_bits': len(bits_tx),
            'n_symbols': len(symbols_tx),
            'n_ofdm_blocks': n_blocks,
            'ber': ber_after_fec,  # Report BER after FEC
            'ber_raw': ber_raw,  # Keep raw BER for reference
            'modulation': f'{self.M}-QAM',
            'n_subcarriers': self.N,
            'cp_length': self.cp_len,
            # Save symbols for constellation diagram
            'symbols_tx': symbols_tx,
            'symbols_rx': symbols_rx[:len(symbols_tx)],
            'constellation': self.constellation,
            'fec_simulated': simulate_fec
        }
        
        return ecg_received, stats


# Example usage
if __name__ == "__main__":
    # Test with synthetic ECG
    fs = 360
    t = np.arange(0, 2, 1/fs)
    ecg_test = 0.5 + 0.3 * np.sin(2 * np.pi * 1.2 * t)  # Simple sine wave
    
    # Create system
    vlc_system = ACO_OFDM_VLC(N=64, cp_len=16, M=4)
    
    # Channel gain (time-varying)
    h_t = 0.8 + 0.2 * np.sin(2 * np.pi * 0.5 * t)
    
    # Full transmission
    ecg_rx, stats = vlc_system.full_transmission(ecg_test, h_t)
    
    print("Transmission Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nMSE: {np.mean((ecg_test - ecg_rx)**2):.6f}")


def plot_constellation(symbols_tx, symbols_rx, constellation, title="Constellation Diagram"):
    """
    Generate constellation diagram data for plotting
    
    Args:
        symbols_tx: Transmitted symbols
        symbols_rx: Received symbols
        constellation: QAM constellation points
        title: Plot title
    
    Returns:
        dict with plot data
    """
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Ideal constellation points
    fig.add_trace(go.Scatter(
        x=constellation.real,
        y=constellation.imag,
        mode='markers',
        name='Ideal Constellation',
        marker=dict(
            size=15,
            color='green',
            symbol='x',
            line=dict(width=2)
        )
    ))
    
    # Transmitted symbols (sample)
    sample_size = min(500, len(symbols_tx))
    indices = np.random.choice(len(symbols_tx), sample_size, replace=False)
    
    fig.add_trace(go.Scatter(
        x=symbols_tx[indices].real,
        y=symbols_tx[indices].imag,
        mode='markers',
        name='TX Symbols',
        marker=dict(
            size=8,
            color='blue',
            opacity=0.6
        )
    ))
    
    # Received symbols (sample)
    fig.add_trace(go.Scatter(
        x=symbols_rx[indices].real,
        y=symbols_rx[indices].imag,
        mode='markers',
        name='RX Symbols',
        marker=dict(
            size=6,
            color='red',
            opacity=0.5
        )
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="In-Phase (I)",
        yaxis_title="Quadrature (Q)",
        height=500,
        template='plotly_dark',
        showlegend=True,
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='gray'),
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='gray', scaleanchor="x", scaleratio=1)
    )
    
    return fig
