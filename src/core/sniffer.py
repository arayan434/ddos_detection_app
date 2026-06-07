import time
import numpy as np
from scapy.all import sniff, IP, TCP, UDP
from typing import Dict, List
from src.utils.logger import log


class NetworkFlow:
    """Агрегує пакети та розраховує 48 статистичних ознак потоку."""

    def __init__(self, flow_id: str, proto: int):
        self.flow_id = flow_id
        self.protocol = proto
        self.start_time = time.time()
        self.last_packet_time = self.start_time

        self.fwd_packets = 0
        self.bwd_packets = 0
        self.fwd_bytes = 0
        self.bwd_bytes = 0
        self.fwd_header_len = 0

        self.fwd_pkt_lengths: List[int] = []
        self.bwd_pkt_lengths: List[int] = []
        self.all_pkt_lengths: List[int] = []

        self.fwd_iat: List[float] = []
        self.bwd_iat: List[float] = []
        self.all_iat: List[float] = []

        self.last_fwd_time = 0.0
        self.last_bwd_time = 0.0

        # Прапорці TCP
        self.flags = {'FIN': 0, 'SYN': 0, 'PSH': 0, 'URG': 0, 'CWE': 0, 'ECE': 0}
        self.fwd_psh = 0
        self.bwd_psh = 0
        self.fwd_urg = 0
        self.bwd_urg = 0

        self.init_win_fwd = 0
        self.init_win_bwd = 0
        self.min_seg_fwd = -1

    def add_packet(self, pkt, direction: str, current_time: float):
        payload_len = len(pkt[IP].payload)
        header_len = len(pkt[IP]) - payload_len

        # Фіксуємо IAT (Inter-Arrival Time)
        if self.last_packet_time > 0 and self.last_packet_time != self.start_time:
            self.all_iat.append(current_time - self.last_packet_time)

        self.last_packet_time = current_time
        self.all_pkt_lengths.append(payload_len)

        # Обробка TCP специфіки
        if TCP in pkt:
            tcp_flags = pkt[TCP].flags
            if 'F' in tcp_flags: self.flags['FIN'] += 1
            if 'S' in tcp_flags: self.flags['SYN'] += 1
            if 'P' in tcp_flags: self.flags['PSH'] += 1
            if 'U' in tcp_flags: self.flags['URG'] += 1
            if 'C' in tcp_flags: self.flags['CWE'] += 1
            if 'E' in tcp_flags: self.flags['ECE'] += 1

            if direction == 'fwd':
                if 'P' in tcp_flags: self.fwd_psh += 1
                if 'U' in tcp_flags: self.fwd_urg += 1
                if self.init_win_fwd == 0: self.init_win_fwd = pkt[TCP].window
                if self.min_seg_fwd == -1 or header_len < self.min_seg_fwd:
                    self.min_seg_fwd = header_len
            else:
                if 'P' in tcp_flags: self.bwd_psh += 1
                if 'U' in tcp_flags: self.bwd_urg += 1
                if self.init_win_bwd == 0: self.init_win_bwd = pkt[TCP].window

        if direction == 'fwd':
            self.fwd_packets += 1
            self.fwd_bytes += payload_len
            self.fwd_header_len += header_len
            self.fwd_pkt_lengths.append(payload_len)

            if self.last_fwd_time > 0:
                self.fwd_iat.append(current_time - self.last_fwd_time)
            self.last_fwd_time = current_time
        else:
            self.bwd_packets += 1
            self.bwd_bytes += payload_len
            self.bwd_pkt_lengths.append(payload_len)

            if self.last_bwd_time > 0:
                self.bwd_iat.append(current_time - self.last_bwd_time)
            self.last_bwd_time = current_time

    def _safe_stat(self, data: List, stat_type: str):
        if not data: return 0.0
        if stat_type == 'max': return float(np.max(data))
        if stat_type == 'min': return float(np.min(data))
        if stat_type == 'mean': return float(np.mean(data))
        if stat_type == 'std': return float(np.std(data))
        if stat_type == 'sum': return float(np.sum(data))
        if stat_type == 'var': return float(np.var(data))
        return 0.0

    def extract_features(self) -> np.ndarray:
        duration = max(self.last_packet_time - self.start_time, 0.000001)
        duration_micro = duration * 1e6

        features = {
            'Protocol': self.protocol,
            'Flow Duration': duration_micro,
            'Total Fwd Packets': self.fwd_packets,
            'Total Backward Packets': self.bwd_packets,

            'Fwd Packet Length Max': self._safe_stat(self.fwd_pkt_lengths, 'max'),
            'Fwd Packet Length Min': self._safe_stat(self.fwd_pkt_lengths, 'min'),
            'Fwd Packet Length Std': self._safe_stat(self.fwd_pkt_lengths, 'std'),

            'Bwd Packet Length Max': self._safe_stat(self.bwd_pkt_lengths, 'max'),
            'Bwd Packet Length Min': self._safe_stat(self.bwd_pkt_lengths, 'min'),
            'Bwd Packet Length Mean': self._safe_stat(self.bwd_pkt_lengths, 'mean'),
            'Bwd Packet Length Std': self._safe_stat(self.bwd_pkt_lengths, 'std'),

            'Flow Bytes/s': (self.fwd_bytes + self.bwd_bytes) / duration,
            'Flow Packets/s': (self.fwd_packets + self.bwd_packets) / duration,

            'Flow IAT Mean': self._safe_stat(self.all_iat, 'mean') * 1e6,
            'Flow IAT Min': self._safe_stat(self.all_iat, 'min') * 1e6,
            'Fwd IAT Min': self._safe_stat(self.fwd_iat, 'min') * 1e6,

            'Bwd IAT Total': self._safe_stat(self.bwd_iat, 'sum') * 1e6,
            'Bwd IAT Mean': self._safe_stat(self.bwd_iat, 'mean') * 1e6,
            'Bwd IAT Max': self._safe_stat(self.bwd_iat, 'max') * 1e6,
            'Bwd IAT Min': self._safe_stat(self.bwd_iat, 'min') * 1e6,

            'Fwd PSH Flags': self.fwd_psh,
            'Bwd PSH Flags': self.bwd_psh,
            'Fwd URG Flags': self.fwd_urg,
            'Bwd URG Flags': self.bwd_urg,
            'Fwd Header Length': self.fwd_header_len,
            'Bwd Packets/s': self.bwd_packets / duration,

            'Packet Length Std': self._safe_stat(self.all_pkt_lengths, 'std'),
            'Packet Length Variance': self._safe_stat(self.all_pkt_lengths, 'var'),

            'FIN Flag Count': self.flags['FIN'],
            'SYN Flag Count': self.flags['SYN'],
            'PSH Flag Count': self.flags['PSH'],
            'URG Flag Count': self.flags['URG'],
            'CWE Flag Count': self.flags['CWE'],
            'ECE Flag Count': self.flags['ECE'],

            'Down/Up Ratio': (self.bwd_packets / self.fwd_packets) if self.fwd_packets > 0 else 0,

            # Апроксимація Bulk характеристик (складна імплементація для реал-тайму)
            'Fwd Avg Bytes/Bulk': 0.0,
            'Fwd Avg Packets/Bulk': 0.0,
            'Fwd Avg Bulk Rate': 0.0,
            'Bwd Avg Bytes/Bulk': 0.0,
            'Bwd Avg Packets/Bulk': 0.0,
            'Bwd Avg Bulk Rate': 0.0,

            'Init_Win_bytes_forward': self.init_win_fwd,
            'Init_Win_bytes_backward': self.init_win_bwd,
            'min_seg_size_forward': self.min_seg_fwd if self.min_seg_fwd != -1 else 0,

            # Апроксимація Active/Idle
            'Active Mean': 0.0,
            'Active Std': 0.0,
            'Active Max': 0.0,
            'Idle Std': 0.0
        }

        # Суворий порядок згідно з препроцесором
        expected_columns = [
            'Protocol', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Std',
            'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
            'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
            'Flow IAT Min', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Max',
            'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
            'Fwd Header Length', 'Bwd Packets/s', 'Packet Length Std', 'Packet Length Variance',
            'FIN Flag Count', 'SYN Flag Count', 'PSH Flag Count', 'URG Flag Count',
            'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Fwd Avg Bytes/Bulk',
            'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk',
            'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Init_Win_bytes_forward',
            'Init_Win_bytes_backward', 'min_seg_size_forward', 'Active Mean', 'Active Std',
            'Active Max', 'Idle Std'
        ]

        vector = [features[col] for col in expected_columns]
        return np.array(vector, dtype=np.float32)


class LiveTrafficSniffer:
    def __init__(self, interface: str):
        self.interface = interface
        self.active_flows: Dict[str, NetworkFlow] = {}

    def _generate_flow_id(self, pkt) -> str:
        if IP not in pkt: return ""
        src, dst, proto = pkt[IP].src, pkt[IP].dst, pkt[IP].proto
        sport, dport = 0, 0
        if TCP in pkt:
            sport, dport = pkt[TCP].sport, pkt[TCP].dport
        elif UDP in pkt:
            sport, dport = pkt[UDP].sport, pkt[UDP].dport

        return f"{src}:{sport}-{dst}:{dport}-{proto}" if src > dst else f"{dst}:{dport}-{src}:{sport}-{proto}"

    def _process_packet(self, pkt):
        if IP not in pkt: return
        flow_id = self._generate_flow_id(pkt)
        if not flow_id: return

        current_time = time.time()
        if flow_id not in self.active_flows:
            self.active_flows[flow_id] = NetworkFlow(flow_id, pkt[IP].proto)

        direction = 'fwd' if pkt[IP].src == flow_id.split(':')[0] else 'bwd'
        self.active_flows[flow_id].add_packet(pkt, direction, current_time)

    def get_flow_features(self, flow_id: str) -> np.ndarray:
        if flow_id in self.active_flows:
            return self.active_flows[flow_id].extract_features()
        return None

    def start(self):
        """Запускає процес перехоплення пакетів."""
        log.info(f"Початок перехоплення трафіку на інтерфейсі {self.interface}...")
        try:
            # store=False критично важливо, щоб пакети не накопичувалися в ОЗП
            sniff(
                iface=self.interface,
                prn=self._process_packet,
                store=False,
                filter="ip"  # Захоплюємо лише IP-трафік
            )
        except PermissionError:
            log.error("Помилка доступу: Для перехоплення мережевого трафіку необхідні права адміністратора!")
        except Exception as e:
            log.error(f"Критична помилка сніфера: {e}")
            log.warning("Якщо ви на Windows, переконайтеся, що у вас встановлено Npcap (входить до складу Wireshark).")