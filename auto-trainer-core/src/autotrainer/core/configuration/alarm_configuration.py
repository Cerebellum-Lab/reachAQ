
from dataclasses import dataclass, field

from autotrainer.core.configuration.animal_thrash_config import AnimalThrashAlarmConfig
from autotrainer.core.configuration.device_comm_alarm_config import DeviceCommAlarmConfig
from autotrainer.core.configuration.animal_presence_configuration import GlobalAnimalPresenceConfig
from autotrainer.core.configuration.autoclamp_evasion_config import AnimalEvasionAlarmConfig
from autotrainer.core.configuration.detector import DetectorConfig
from autotrainer.core.configuration.external_doors_monitor_configuration import ExternalDoorsAlarmConfig
from autotrainer.core.configuration.system_fault_config import SystemFaultConfig
from autotrainer.core.configuration.system_maintenance_config import SystemMaintenanceConfig


@dataclass
class EmergencyAlarmConfiguration(DetectorConfig):

    # 1st possible alarm condition
    animal_thrashing: AnimalThrashAlarmConfig = field(default_factory=AnimalThrashAlarmConfig)

    # 2nd possible alarm condition

    # 3rd
    external_doors: ExternalDoorsAlarmConfig = field(default_factory=ExternalDoorsAlarmConfig)

    # 4rd
    global_animal_presence: GlobalAnimalPresenceConfig = field(default_factory=GlobalAnimalPresenceConfig)

    # 5th
    device_comm_error: DeviceCommAlarmConfig = field(default_factory=DeviceCommAlarmConfig)

    # 6th
    system_maintenance: SystemMaintenanceConfig = field(default_factory=SystemMaintenanceConfig)

    # 7th
    system_fault: SystemFaultConfig = field(default_factory=SystemFaultConfig)

    # 8th
    animal_evasion: AnimalEvasionAlarmConfig = field(default_factory=AnimalEvasionAlarmConfig)
