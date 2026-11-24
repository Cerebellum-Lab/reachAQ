
import json

from datetime import datetime, date, time


class SystemConfigurationJSONEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, (time, datetime, date)):
            return obj.isoformat()
        return super().default(obj)
