from scapy.arch.windows import get_windows_if_list

print("Доступні мережеві інтерфейси в системі:\n" + "-"*50)
interfaces = get_windows_if_list()

for iface in interfaces:
    print(f"Назва (ОС):  {iface.get('name', 'Невідомо')}")
    print(f"Опис:        {iface.get('description', 'Невідомо')}")
    print(f"Scapy GUID:  {iface.get('guid', 'Невідомо')}")
    print(f"IP-адреси:   {iface.get('ips', [])}")
    print("-" * 50)