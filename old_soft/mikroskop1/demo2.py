import time
from pylinkam import interface, sdk

wrapper = sdk.SDKWrapper()
# connection = wrapper.connect()

with wrapper.connect() as connection:
    print(f"Name: {connection.get_controller_name()}")