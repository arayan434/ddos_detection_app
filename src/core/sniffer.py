import time
import threading
import numpy as np
from scapy.all import sniff, IP, TCP, UDP
from typing import Dict, List, Tuple
from src.utils.logger import log


class NetworkFlow:
    """Агрегує пакети та розраховує 48 статистичних ознак потоку з евристиками."""

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

        self.flags = {'FIN': 0, 'SYN': 0, 'PSH': 0, 'URG': 0, 'CWE': 0, 'ECE': 0}
        self.fwd_psh = 0
        self.bwd_psh = 0
        self.fwd_urg = 0
        self.bwd_urg = 0

        self.init_win_fwd = 0
        self.init_win_bwd = 0
        self.min_seg_fwd = -1

        # Змінні для розрахунку Active / Idle
        self.idle_threshold = 1.0  # Простій більше 1 секунди вважається Idle
        self.current_active_start = self.start_time
        self.active_times: List[float] = []
        self.idle_times: List[float] = []

    def add_packet(self, pkt, direction: str, current_time: float):
        payload_len = len(pkt[IP].payload)
        header_len = len(pkt[IP]) - payload_len

        # Фіксуємо IAT та розраховуємо фази активності
        if self.last_packet_time > 0 and self.last_packet_time != self.start_time:
            iat = current_time - self.last_packet_time
            self.all_iat.append(iat)

            # Евристика Active/Idle
            if iat > self.idle_threshold:
                self.idle_times.append(iat * 1e6)  # В мікросекундах
                active_duration = (self.last_packet_time - self.current_active_start) * 1e6
                if active_duration > 0:
                    self.active_times.append(active_duration)
                self.current_active_start = current_time

        self.last_packet_time = current_time
        self.all_pkt_lengths.append(payload_len)

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

        # Динамічний розрахунок поточної фази активності
        current_active = (self.last_packet_time - self.current_active_start) * 1e6
        temp_active = self.active_times + [current_active] if current_active > 0 else self.active_times

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

            'Fwd Avg Bytes/Bulk': 0.0,
            'Fwd Avg Packets/Bulk': 0.0,
            'Fwd Avg Bulk Rate': 0.0,
            'Bwd Avg Bytes/Bulk': 0.0,
            'Bwd Avg Packets/Bulk': 0.0,
            'Bwd Avg Bulk Rate': 0.0,

            'Init_Win_bytes_forward': self.init_win_fwd,
            'Init_Win_bytes_backward': self.init_win_bwd,
            'min_seg_size_forward': self.min_seg_fwd if self.min_seg_fwd != -1 else 0,

            # Заміна нулів на реальні евристики
            'Active Mean': self._safe_stat(temp_active, 'mean'),
            'Active Std': self._safe_stat(temp_active, 'std'),
            'Active Max': self._safe_stat(temp_active, 'max'),
            'Idle Std': self._safe_stat(self.idle_times, 'std')
        }

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

        # Архітектурні запобіжники потокобезпеки
        self.lock = threading.Lock()
        self.flow_timeout = 60.0  # Секунди простою до видалення
        self.max_concurrent_flows = 50000
        self.last_cleanup = time.time()

    def _generate_flow_id(self, pkt) -> str:
        if IP not in pkt: return ""
        src, dst, proto = pkt[IP].src, pkt[IP].dst, pkt[IP].proto
        sport, dport = 0, 0
        if TCP in pkt:
            sport, dport = pkt[TCP].sport, pkt[TCP].dport
        elif UDP in pkt:
            sport, dport = pkt[UDP].sport, pkt[UDP].dport

        return f"{src}:{sport}-{dst}:{dport}-{proto}" if src > dst else f"{dst}:{dport}-{src}:{sport}-{proto}"

    def _cleanup_inactive_flows(self, current_time: float):
        """Звільняє пам'ять від розірваних або завислих з'єднань."""
        if current_time - self.last_cleanup < 10.0:
            return

        self.last_cleanup = current_time
        expired_flows = [
            f_id for f_id, flow in self.active_flows.items()
            if current_time - flow.last_packet_time > self.flow_timeout
        ]

        for f_id in expired_flows:
            del self.active_flows[f_id]

        if expired_flows:
            log.debug(f"Очищення: видалено {len(expired_flows)} потоків. Активних: {len(self.active_flows)}")

    def _process_packet(self, pkt):
        if IP not in pkt: return
        flow_id = self._generate_flow_id(pkt)
        if not flow_id: return

        current_time = time.time()
        direction = 'fwd' if pkt[IP].src == flow_id.split(':')[0] else 'bwd'

        with self.lock:
            self._cleanup_inactive_flows(current_time)

            if flow_id not in self.active_flows:
                if len(self.active_flows) >= self.max_concurrent_flows:
                    return  # Дроп пакету для захисту ОЗП
                self.active_flows[flow_id] = NetworkFlow(flow_id, pkt[IP].proto)

            self.active_flows[flow_id].add_packet(pkt, direction, current_time)

    def get_all_active_flow_ids(self) -> List[str]:
        """Потокобезпечне отримання ідентифікаторів."""
        with self.lock:
            return list(self.active_flows.keys())

    def get_flow_stats_and_features(self, flow_id: str) -> Tuple[dict, np.ndarray]:
        """Єдиний безпечний метод читання стану потоку для ядра."""
        with self.lock:
            flow = self.active_flows.get(flow_id)
            if not flow: return None, None

            stats = {
                "protocol": flow.protocol,
                "packets": flow.fwd_packets + flow.bwd_packets,
                "bytes": flow.fwd_bytes + flow.bwd_bytes,
                "last_packet_time": flow.last_packet_time
            }
            features = flow.extract_features()
            return stats, features

    def start(self):
        log.info(f"Початок перехоплення трафіку на інтерфейсі {self.interface}...")
        try:
            sniff(
                iface=self.interface,
                prn=self._process_packet,
                store=False,
                filter="ip"
            )
        except Exception as e:
            log.error(f"Критична помилка сніфера: {e}")