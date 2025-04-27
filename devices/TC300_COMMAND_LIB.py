import os
from ctypes import *


#region import dll functions
cwd = os.getcwd()
print(cwd)
TC300Lib = cdll.LoadLibrary(os.path.join(cwd, "devices", "TC300COMMANDLIB_win64.dll"))
# TC300Lib = cdll.LoadLibrary("c:/Users/LAB218/Documents/mikroskop2/TC300COMMANDLIB_win64.dll")



"""common command
"""
List = TC300Lib.List
List.restype = c_int
List.argtypes = [c_char_p, c_int]

Open = TC300Lib.Open
Open.restype = c_int
Open.argtypes = [c_char_p,c_int,c_int]

IsOpen = TC300Lib.IsOpen
IsOpen.restype = c_int
IsOpen.argtypes = [c_char_p]

Close = TC300Lib.Close
Close.restype = c_int
Close.argtypes = [c_int]

GetHandle = TC300Lib.GetHandle
GetHandle.restype = c_int
GetHandle.argtypes = [c_char_p]

"""device command
"""
GetId = TC300Lib.GetId
GetId.restype = c_int
GetId.argtypes = [c_int,c_char_p]

GetStatus = TC300Lib.GetStatus
GetStatus.restype = c_int
GetStatus.argtypes = [c_int,POINTER(c_ubyte)]

EnableChannel = TC300Lib.EnableChannel
EnableChannel.restype = c_int
EnableChannel.argtypes = [c_int,c_int,c_ubyte]

GetChannels = TC300Lib.GetChannels
GetChannels.restype = c_int
GetChannels.argtypes = [c_int,POINTER(c_int)]

CopyParameters = TC300Lib.CopyParameters
CopyParameters.restype = c_int
CopyParameters.argtypes = [c_int]

SetChannels = TC300Lib.SetChannels
SetChannels.restype = c_int
SetChannels.argtypes = [c_int,c_int]

GetT0Constant = TC300Lib.GetT0Constant
GetT0Constant.restype = c_int
GetT0Constant.argtypes = [c_int,c_int,POINTER(c_float)]

SetT0Constant = TC300Lib.SetT0Constant
SetT0Constant.restype = c_int
SetT0Constant.argtypes = [c_int,c_int,c_float]

GetR0Constant = TC300Lib.GetR0Constant
GetR0Constant.restype = c_int
GetR0Constant.argtypes = [c_int,c_int,POINTER(c_float)]

SetR0Constant = TC300Lib.SetR0Constant
SetR0Constant.restype = c_int
SetR0Constant.argtypes = [c_int,c_int,c_float]

GetHartAConstant = TC300Lib.GetHartAConstant
GetHartAConstant.restype = c_int
GetHartAConstant.argtypes = [c_int,c_int,POINTER(c_float)]

SetHartAConstant = TC300Lib.SetHartAConstant
SetHartAConstant.restype = c_int
SetHartAConstant.argtypes = [c_int,c_int,c_float]

GetHartBConstant = TC300Lib.GetHartBConstant
GetHartBConstant.restype = c_int
GetHartBConstant.argtypes = [c_int,c_int,POINTER(c_float)]

SetHartBConstant = TC300Lib.SetHartBConstant
SetHartBConstant.restype = c_int
SetHartBConstant.argtypes = [c_int,c_int,c_float]

GetHartCConstant = TC300Lib.GetHartCConstant
GetHartCConstant.restype = c_int
GetHartCConstant.argtypes = [c_int,c_int,POINTER(c_float)]

SetHartCConstant = TC300Lib.SetHartCConstant
SetHartCConstant.restype = c_int
SetHartCConstant.argtypes = [c_int,c_int,c_float]

GetDarkStatus = TC300Lib.GetDarkStatus
GetDarkStatus.restype = c_int
GetDarkStatus.argtypes = [c_int,POINTER(c_int)]

SetDarkStatus = TC300Lib.SetDarkStatus
SetDarkStatus.restype = c_int
SetDarkStatus.argtypes = [c_int,c_int]

GetQuietMode = TC300Lib.GetQuietMode
GetQuietMode.restype = c_int
GetQuietMode.argtypes = [c_int,POINTER(c_int)]

SetQuietMode = TC300Lib.SetQuietMode
SetQuietMode.restype = c_int
SetQuietMode.argtypes = [c_int,c_int]

GetActualTemperature = TC300Lib.GetActualTemperature
GetActualTemperature.restype = c_int
GetActualTemperature.argtypes = [c_int,c_int,POINTER(c_float)]

SetTargetTemperature = TC300Lib.SetTargetTemperature
SetTargetTemperature.restype = c_int
SetTargetTemperature.argtypes = [c_int,c_int,c_float]

GetTargetTemperature = TC300Lib.GetTargetTemperature
GetTargetTemperature.restype = c_int
GetTargetTemperature.argtypes = [c_int,c_int,POINTER(c_float)]

SetMaxTemperature = TC300Lib.SetMaxTemperature
SetMaxTemperature.restype = c_int
SetMaxTemperature.argtypes = [c_int,c_int,c_float]

GetMaxTemperature = TC300Lib.GetMaxTemperature
GetMaxTemperature.restype = c_int
GetMaxTemperature.argtypes = [c_int,c_int,POINTER(c_float)]

SetMinTemperature = TC300Lib.SetMinTemperature
SetMinTemperature.restype = c_int
SetMinTemperature.argtypes = [c_int,c_int,c_float]

GetMinTemperature = TC300Lib.GetMinTemperature
GetMinTemperature.restype = c_int
GetMinTemperature.argtypes = [c_int,c_int,POINTER(c_float)]

GetOutputVoltage = TC300Lib.GetOutputVoltage
GetOutputVoltage.restype = c_int
GetOutputVoltage.argtypes = [c_int,c_int,POINTER(c_float)]

GetTargetOutputCurrent = TC300Lib.GetTargetOutputCurrent
GetTargetOutputCurrent.restype = c_int
GetTargetOutputCurrent.argtypes = [c_int,c_int,POINTER(c_float)]

GetOutputCurrent = TC300Lib.GetOutputCurrent
GetOutputCurrent.restype = c_int
GetOutputCurrent.argtypes = [c_int,c_int,POINTER(c_float)]

SetOutputCurrent = TC300Lib.SetOutputCurrent
SetOutputCurrent.restype = c_int
SetOutputCurrent.argtypes = [c_int,c_int,c_float]

GetMaxVoltage = TC300Lib.GetMaxVoltage
GetMaxVoltage.restype = c_int
GetMaxVoltage.argtypes = [c_int,c_int,POINTER(c_float)]

SetMaxVoltage = TC300Lib.SetMaxVoltage
SetMaxVoltage.restype = c_int
SetMaxVoltage.argtypes = [c_int,c_int,c_float]

GetMaxCurrent = TC300Lib.GetMaxCurrent
GetMaxCurrent.restype = c_int
GetMaxCurrent.argtypes = [c_int,c_int,POINTER(c_float)]

SetMaxCurrent = TC300Lib.SetMaxCurrent
SetMaxCurrent.restype = c_int
SetMaxCurrent.argtypes = [c_int,c_int,c_float]

GetMode = TC300Lib.GetMode
GetMode.restype = c_int
GetMode.argtypes = [c_int,c_int,POINTER(c_int)]

SetMode = TC300Lib.SetMode
SetMode.restype = c_int
SetMode.argtypes = [c_int,c_int,c_int]

GetSensorOffset = TC300Lib.GetSensorOffset
GetSensorOffset.restype = c_int
GetSensorOffset.argtypes = [c_int,c_int,POINTER(c_float)]

SetSensorOffset = TC300Lib.SetSensorOffset
SetSensorOffset.restype = c_int
SetSensorOffset.argtypes = [c_int,c_int,c_float]

GetSensorType = TC300Lib.GetSensorType
GetSensorType.restype = c_int
GetSensorType.argtypes = [c_int,c_int,POINTER(c_int)]

SetSensorType = TC300Lib.SetSensorType
SetSensorType.restype = c_int
SetSensorType.argtypes = [c_int,c_int,c_int]

GetSensorParameter = TC300Lib.GetSensorParameter
GetSensorParameter.restype = c_int
GetSensorParameter.argtypes = [c_int,c_int,POINTER(c_int)]

SetSensorParameter = TC300Lib.SetSensorParameter
SetSensorParameter.restype = c_int
SetSensorParameter.argtypes = [c_int,c_int,c_int]

GetNtcBeta = TC300Lib.GetNtcBeta
GetNtcBeta.restype = c_int
GetNtcBeta.argtypes = [c_int,c_int,POINTER(c_float)]

SetNtcBeta = TC300Lib.SetNtcBeta
SetNtcBeta.restype = c_int
SetNtcBeta.argtypes = [c_int,c_int,c_float]

GetExtBeta = TC300Lib.GetExtBeta
GetExtBeta.restype = c_int
GetExtBeta.argtypes = [c_int,c_int,POINTER(c_float)]

SetExtBeta = TC300Lib.SetExtBeta
SetExtBeta.restype = c_int
SetExtBeta.argtypes = [c_int,c_int,c_float]





GetTriggerMode = TC300Lib.GetTriggerMode
GetTriggerMode.restype = c_int
GetTriggerMode.argtypes = [c_int,c_int,POINTER(c_int)]

SetTriggerMode = TC300Lib.SetTriggerMode
SetTriggerMode.restype = c_int
SetTriggerMode.argtypes = [c_int,c_int,c_int]

GetBrightness = TC300Lib.GetBrightness
GetBrightness.restype = c_int
GetBrightness.argtypes = [c_int,POINTER(c_int)]

SetBrightness = TC300Lib.SetBrightness
SetBrightness.restype = c_int
SetBrightness.argtypes = [c_int,c_int]

GetKnobState = TC300Lib.GetKnobState
GetKnobState.restype = c_int
GetKnobState.argtypes = [c_int,POINTER(c_int)]

SetKnobState = TC300Lib.SetKnobState
SetKnobState.restype = c_int
SetKnobState.argtypes = [c_int,c_int]

GetPIDParameter = TC300Lib.GetPIDParameter
GetPIDParameter.restype = c_int
GetPIDParameter.argtypes = [c_int,c_int,POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_int)]

GetPIDParameterP = TC300Lib.GetPIDParameterP
GetPIDParameterP.restype = c_int
GetPIDParameterP.argtypes = [c_int,c_int,POINTER(c_float)]

SetPIDParameterP = TC300Lib.SetPIDParameterP
SetPIDParameterP.restype = c_int
SetPIDParameterP.argtypes = [c_int,c_int,c_float]

GetPIDParameterI = TC300Lib.GetPIDParameterI
GetPIDParameterI.restype = c_int
GetPIDParameterI.argtypes = [c_int,c_int,POINTER(c_float)]

SetPIDParameterI = TC300Lib.SetPIDParameterI
SetPIDParameterI.restype = c_int
SetPIDParameterI.argtypes = [c_int,c_int,c_float]

GetPIDParameterD = TC300Lib.GetPIDParameterD
GetPIDParameterD.restype = c_int
GetPIDParameterD.argtypes = [c_int,c_int,POINTER(c_float)]

SetPIDParameterD = TC300Lib.SetPIDParameterD
SetPIDParameterD.restype = c_int
SetPIDParameterD.argtypes = [c_int,c_int,c_float]

GetPIDParameterPeriod = TC300Lib.GetPIDParameterPeriod
GetPIDParameterPeriod.restype = c_int
GetPIDParameterPeriod.argtypes = [c_int,c_int,POINTER(c_int)]

SetPIDParameterPeriod = TC300Lib.SetPIDParameterPeriod
SetPIDParameterPeriod.restype = c_int
SetPIDParameterPeriod.argtypes = [c_int,c_int,c_int]

LoadDefaultPIDParameter = TC300Lib.LoadDefaultPIDParameter
LoadDefaultPIDParameter.restype = c_int
LoadDefaultPIDParameter.argtypes = [c_int,c_int]

GetMonitorMessage = TC300Lib.GetMonitorMessage
GetMonitorMessage.restype = c_int
GetMonitorMessage.argtypes = [c_int,POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_int),POINTER(c_int),POINTER(c_int),
                              POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_int),POINTER(c_int),POINTER(c_int)]

GetErrorMessage = TC300Lib.GetErrorMessage
GetErrorMessage.restype = c_int
GetErrorMessage.argtypes = [c_int,POINTER(c_ubyte)]

GetWarningMessage = TC300Lib.GetWarningMessage
GetWarningMessage.restype = c_int
GetWarningMessage.argtypes = [c_int,POINTER(c_ubyte)]

LoadFactoryParameter = TC300Lib.LoadFactoryParameter
LoadFactoryParameter.restype = c_int
LoadFactoryParameter.argtypes = [c_int]

#region command for TC300
def TC300ListDevices():
    """ List all connected TC300 devices
    Returns: 
       The TC300 device list, each deice item is [serialNumber]
    """
    str = create_string_buffer(1024, '\0') 
    result = List(str,1024)
    devicesStr = str.raw.decode("utf-8").rstrip('\x00').split(',')
    length = len(devicesStr)
    i = 0
    devices = []
    devInfo = ["",""]
    while(i < length):
        str = devicesStr[i]
        if (i % 2 == 0):
            if str != '':
                devInfo[0] = str
            else:
                i+=1
        else:
            devInfo[1] = str
            devices.append(devInfo.copy())
        i+=1
    return devices


def TC300Open(serialNo, nBaud, timeout):
    """ Open TC300 device
    Args:
        serialNo: serial number of TC300 device
        nBaud: bit per second of port
        timeout: set timeout value in (s)
    Returns: 
        non-negative number: hdl number returned Successful; negative number: failed.
    """
    return Open(serialNo.encode('utf-8'), nBaud, timeout)

def TC300IsOpen(serialNo):
    """ Check opened status of TC300 device
    Args:
        serialNo: serial number of TC300 device
    Returns: 
        0: TC300 device is not opened; 1: TC300 device is opened.
    """
    return IsOpen(serialNo.encode('utf-8'))

def TC300GetHandle(serialNo):
    """ get handle of port
    Args:
        serialNo: serial number of the device to be checked.
    Returns: 
        -1:no handle  non-negtive number: handle.
    """
    return GetHandle(serialNo.encode('utf-8'))

def TC300Close(hdl):
    """ Close opened TC300 device
    Args:
        hdl: the handle of opened TC300 device
    Returns: 
        0: Success; negative number: failed.
    """
    return Close(hdl)

def TC300GetId(hdl, id):
    """ get the TC300 id
    Args:
        hdl: the handle of opened TC300 device
        value: the model number, hardware and firmware versions
    Returns: 
        0: Success; negative number: failed.
    """
    idStr = create_string_buffer(1024,'\0')
    ret = GetId(hdl,idStr)
    id[0] = idStr.raw.decode("utf-8","ignore").rstrip('\x00').replace(">\r\n", "")
    return ret

def TC300GetStatus(hdl, status):
    """  Get device status
    Args:
        hdl: handle of port.
        status: 
          Device state define 	8bit
          bit0 = 0 or 1	chan1 disabled / enabled
          bit1 = 0 or 1	chan2 disabled / enabled
          bit2 = 1	normal
          bit3 = 1	warning
          bit4 = 1	error
          bit5	Reserved
          bit6	Reserved
          bit7	Reserved
    Returns: 
          0: Success; negative number: failed.
    """
    sta = c_ubyte(0) 
    ret = GetStatus(hdl,sta)
    status[0] = sta.value
    return ret

def TC300EnableChannel(hdl, channel, status):
    """ Set channel enable state 
    Args:
       hdl: handle of port.
       channel: channel index, 1 or 2.
       status: 0: Disable; 1: Enable
    Returns: 
        0: Success; negative number: failed.
    """
    return EnableChannel(hdl, channel, status)

def TC300GetChannels(hdl, channels):
    """  Get number of channels
    Args:
         hdl: handle of port.
         channels: channel number: 0: single channel; 1: dual channel.
    Returns: 
         0: Success; negative number: failed
    """
    val = c_int(0)
    ret = GetChannels(hdl,val)
    channels[0] = val.value
    return ret




def TC300CopyParameters(hdl):
    """ Copy channel 1 parameters to channel 2
    Args:
       hdl: handle of port.     
    Returns: 
        0: Success; negative number: failed.
    """
    return CopyParameters(hdl)




def TC300SetChannels(hdl, channels):
    """ Set number of channels
    Args:
        hdl: handle of port.
        channels: channel number: 0: single channel; 1: dual channel.
    Returns: 
        0: Success; negative number: failed.
    """
    return SetChannels(hdl, channels)

def TC300GetR0Constant(hdl, channel, r0):
    """  Get R0 value when sensor type is NTC1 or EXT1
    Args:
         hdl: handle of port.
         channel: channel index, 1 or 2.
         r0: the R0 value for NTC1 or EXT1 sensor.
    Returns: 
         0: Success; negative number: failed
    """
    val = c_float(0)
    ret = GetR0Constant(hdl,channel,val)
    r0[0] = val.value
    return ret

def TC300SetR0Constant(hdl, channel, r0):
    """ Set R0 value when sensor type is NTC1 or EXT1
    Args:
        hdl: handle of port.
        channel: channel index, 1 or 2.
        r0: the R0 value for NTC1 or EXT1 sensor 0~999.
    Returns: 
        0: Success; negative number: failed.
    """
    return SetR0Constant(hdl, channel, r0)

def TC300GetT0Constant(hdl, channel, t0):
    """  Get T0 value when sensor type is NTC1 or EXT1
    Args:
         hdl: handle of port.
         channel: channel index, 1 or 2.
         t0: the T0 value for NTC1 or EXT1 sensor.
    Returns: 
         0: Success; negative number: failed
    """
    val = c_float(0)
    ret = GetT0Constant(hdl,channel,val)
    t0[0] = val.value
    return ret

def TC300SetT0Constant(hdl, channel, t0):
    """ Set T0 value when sensor type is NTC1 or EXT1
    Args:
        hdl: handle of port.
        channel: channel index, 1 or 2.
        t0: the T0 value for NTC1 or EXT1 sensor 0~999.
    Returns: 
        0: Success; negative number: failed.
    """
    return SetT0Constant(hdl, channel, t0)

def TC300GetHartAConstant(hdl, channel, hartA):
    """  Get Hart A value when sensor type is NTC2 or EXT2
    Args:
         hdl: handle of port.
         channel: channel index, 1 or 2.
         hartA: the Hart A value for NTC2 or EXT2 sensor.
    Returns: 
         0: Success; negative number: failed
    """
    val = c_float(0)
    ret = GetHartAConstant(hdl,channel,val)
    hartA[0] = val.value
    return ret

def TC300SetHartAConstant(hdl, channel, hartA):
    """ Set Hart A value when sensor type is NTC2 or EXT2
    Args:
        hdl: handle of port.
        channel: channel index, 1 or 2.
        hartA: the Hart A value for NTC2 or EXT2 sensor -9.9999~9.9999.
    Returns: 
        0: Success; negative number: failed.
    """
    return SetHartAConstant(hdl, channel, hartA)

def TC300GetHartBConstant(hdl, channel, hartB):
    """  Get Hart B value when sensor type is NTC2 or EXT2
    Args:
         hdl: handle of port.
         channel: channel index, 1 or 2.
         hartB: the Hart B value for NTC2 or EXT2 sensor.
    Returns: 
         0: Success; negative number: failed
    """
    val = c_float(0)
    ret = GetHartBConstant(hdl,channel,val)
    hartB[0] = val.value
    return ret

def TC300SetHartBConstant(hdl, channel, hartB):
    """ Set Hart B value when sensor type is NTC2 or EXT2
    Args:
        hdl: handle of port.
        channel: channel index, 1 or 2.
        hartB: the Hart B value for NTC2 or EXT2 sensor -9.9999~9.9999.
    Returns: 
        0: Success; negative number: failed.
    """
    return SetHartBConstant(hdl, channel, hartB)

def TC300GetHartCConstant(hdl, channel, hartC):
    """  Get Hart C value when sensor type is NTC2 or EXT2
    Args:
         hdl: handle of port.
         channel: channel index, 1 or 2.
         hartC: the Hart C value for NTC2 or EXT2 sensor.
    Returns: 
         0: Success; negative number: failed
    """
    val = c_float(0)
    ret = GetHartCConstant(hdl,channel,val)
    hartC[0] = val.value
    return ret

def TC300SetHartCConstant(hdl, channel, hartC):
    """ Set Hart C value when sensor type is NTC2 or EXT2
    Args:
        hdl: handle of port.
        channel: channel index, 1 or 2.
        hartC: the Hart C value for NTC2 or EXT2 sensor -9.9999~9.9999.
    Returns: 
        0: Success; negative number: failed.
    """
    return SetHartCConstant(hdl, channel, hartC)

def TC300GetDarkStatus(hdl, status):
    """ Get dark mode
    Args:
         hdl: handle of port.
         status: dark mode: 0: disable; 1: enable.
    Returns: 
         0: Success; negative number: failed
    """
    val = c_int(0)
    ret = GetDarkStatus(hdl,val)
    status[0] = val.value
    return ret

def TC300SetDarkStatus(hdl, status):
    """ Set dark mode
    Args:
        hdl: handle of port.
        status: dark mode: 0: disable; 1: enable.
    Returns: 
        0: Success; negative number: failed.
    """
    return SetDarkStatus(hdl, status)

def TC300GetQuietMode(hdl, mode):
    """ Get quiet mode
    Args:
         hdl: handle of port.
         mode: quiet mode: 0: disable; 1: enable.
    Returns: 
         0: Success; negative number: failed
    """
    val = c_int(0)
    ret = GetQuietMode(hdl,val)
    mode[0] = val.value
    return ret

def TC300SetQuietMode(hdl, mode):
    """ Set quiet mode
    Args:
        hdl: handle of port.
        mode: quiet mode: 0: disable; 1: enable.
    Returns: 
        0: Success; negative number: failed.
    """
    return SetQuietMode(hdl, mode)


def TC300GetActualTemperature(hdl, channel, temperature):
    """  get actual temperature
    Args:
         hdl: handle of port.
         channel: channel index, 1 or 2.
         temperature: the actual temperature.
    Returns: 
         0: Success; negative number: failed
    """
    val = c_float(0)
    ret = GetActualTemperature(hdl,channel,val)
    temperature[0] = val.value
    return ret

def TC300SetTargetTemperature(hdl, channel, temperature):
    """ Set target temperature
    Args:
        hdl: handle of port.
        channel: channel index, 1 or 2.
        temperature: the target temperature -200~400 °C.
    Returns: 
        0: Success; negative number: failed.
    """
    return SetTargetTemperature(hdl, channel, temperature)

def TC300GetTargetTemperature(hdl, channel, temperature):
    """  Get target temperature
    Args:
         hdl: handle of port.
         channel: channel index, 1 or 2.
         temperature: the target temperature.
    Returns: 
         0: Success; negative number: failed
    """
    val = c_float(0)
    ret = GetTargetTemperature(hdl,channel,val)
    temperature[0] = val.value
    return ret

def TC300SetMaxTemperature(hdl, channel, temperature):
    """ Set max temperature
    Args:
        hdl: handle of port.
        channel: channel index, 1 or 2.
        temperature: the max value of temperature: min temperature ~ 400 °C.
    Returns: 
        0: Success; negative number: failed.
    """
    return SetMaxTemperature(hdl, channel, temperature)

def TC300GetMaxTemperature(hdl, channel, temperature):
    """  Get max temperature
    Args:
         hdl: handle of port.
         channel: channel index, 1 or 2.
         temperature: the max value of temperature.
    Returns: 
         0: Success; negative number: failed
    """
    val = c_float(0)
    ret = GetMaxTemperature(hdl,channel,val)
    temperature[0] = val.value
    return ret

def TC300SetMinTemperature(hdl, channel, temperature):
    """ Set min temperature
    Args:
        hdl: handle of port.
        channel: channel index, 1 or 2.
        temperature: the min value of temperature: -200 ~ max temperature °C.
    Returns: 
        0: Success; negative number: failed.
    """
    return SetMinTemperature(hdl, channel, temperature)

def TC300GetMinTemperature(hdl, channel, temperature):
    """  get min temperature
    Args:
         hdl: handle of port.
         channel: channel index, 1 or 2.
         temperature: the min temperature.
    Returns: 
         0: Success; negative number: failed
    """
    val = c_float(0)
    ret = GetMinTemperature(hdl,channel,val)
    temperature[0] = val.value
    return ret

def TC300GetOutputVoltage(hdl, channel, voltage):
    """  Get output voltage
    Args:
         hdl: handle of port.
         channel: channel index, 1 or 2.
         voltage: the output voltage.
    Returns: 
         0: Success; negative number: failed
    """
    val = c_float(0)
    ret = GetOutputVoltage(hdl,channel,val)
    voltage[0] = val.value
    return ret

def TC300GetTargetOutputCurrent(hdl, channel, current):
    """  Get target output current
    Args:
         hdl: handle of port.
         channel: channel index, 1 or 2.
         current: the target output current.
    Returns: 
         0: Success; negative number: failed
    """
    val = c_float(0)
    ret = GetTargetOutputCurrent(hdl,channel,val)
    current[0] = val.value
    return ret

def TC300GetOutputCurrent(hdl, channel, current):
    """  Get actual output current
    Args:
         hdl: handle of port.
         channel: channel index, 1 or 2.
         current: the actual output current.
    Returns: 
         0: Success; negative number: failed
    """
    val = c_float(0)
    ret = GetOutputCurrent(hdl,channel,val)
    current[0] = val.value
    return ret

def TC300SetOutputCurrent(hdl, channel, current):
    """ Set target output current
    Args:
        hdl: handle of port.
        channel: channel index, 1 or 2.
        current: the target output current -2000~2000mA.
    Returns: 
        0: Success; negative number: failed.
    """
    return SetOutputCurrent(hdl, channel, current)

def TC300GetMaxVoltage(hdl, channel, voltage):
    """  Get max voltage
    Args:
         hdl: handle of port.
         channel: channel index, 1 or 2.
         voltage: the max value of voltage.
    Returns: 
         0: Success; negative number: failed
    """
    val = c_float(0)
    ret = GetMaxVoltage(hdl,channel,val)
    voltage[0] = val.value
    return ret

def TC300SetMaxVoltage(hdl, channel, voltage):
    """ Set max voltage
    Args:
        hdl: handle of port. 
        channel: channel index, 1 or 2.
        voltage: the max value of voltage 0.1 ~ 24.0 V.
    Returns: 
        0: Success; negative number: failed.
    """
    return SetMaxVoltage(hdl, channel, voltage)

def TC300GetMaxCurrent(hdl, channel, current):
    """  Get max current
    Args:
         hdl: handle of port.
         channel: channel index, 1 or 2.
         current: the max value of current(A).
    Returns: 
         0: Success; negative number: failed
    """
    val = c_float(0)
    ret = GetMaxCurrent(hdl,channel,val)
    current[0] = val.value
    return ret

def TC300SetMaxCurrent(hdl, channel, current):
    """ Set max current
    Args:
        hdl: handle of port.
        channel: channel index, 1 or 2.
        current: the max value of current 0~2000 mA.
    Returns: 
        0: Success; negative number: failed.
    """
    return SetMaxCurrent(hdl, channel, current)

def TC300GetMode(hdl, channel, mode):
    """  Get channel mode
    Args:
         hdl: handle of port.
         channel: channel index, 1 or 2.
         mode: channel mode: 0: Heater; 1: Tec; 2: Constant current; 3: Synchronize with Ch1(only for channel 2).
    Returns: 
         0: Success; negative number: failed
    """
    val = c_int(0)
    ret = GetMode(hdl,channel,val)
    mode[0] = val.value
    return ret

def TC300SetMode(hdl, channel, mode):
    """ Set channel mode
    Args:
        hdl: handle of port.
        channel: channel index, 1 or 2.
        mode: channel mode: 0: Heater; 1: Tec; 2: Constant current; 3: Synchronize with Ch1(only for channel 2).
    Returns: 
        0: Success; negative number: failed.
    """
    return SetMode(hdl, channel, mode)

def TC300GetSensorOffset(hdl, channel, offset):
    """  Get sensor offset when sensor type is PT100 or PT1000
    Args:
         hdl: handle of port.
         channel: channel index, 1 or 2.
         offset: the sensor offset
    Returns: 
         0: Success; negative number: failed
    """
    val = c_float(0)
    ret = GetSensorOffset(hdl,channel,val)
    offset[0] = val.value
    return ret

def TC300SetSensorOffset(hdl, channel, offset):
    """ Set sensor offset when sensor type is PT100 or PT1000
    Args:
        hdl: handle of port.
        channel: channel index, 1 or 2.
        offset: the sensor offset -10~10 °C
    Returns: 
        0: Success; negative number: failed.
    """
    return SetSensorOffset(hdl, channel, offset)

def TC300GetSensorType(hdl, channel, sensorType):
    """  Get sensor type
    Args:
         hdl: handle of port.
         channel: channel index, 1 or 2.
         sensorType: sensor type: 0: PT100; 1: PT1000; 2: NTC1; 3: NTC2; 4: Thermo; 5: AD590; 6: EXT1; 7: EXT2.
    Returns: 
         0: Success; negative number: failed
    """
    val = c_int(0)
    ret = GetSensorType(hdl,channel,val)
    sensorType[0] = val.value
    return ret

def TC300SetSensorType(hdl, channel, sensorType):
    """ Set channel mode
    Args:
        hdl: handle of port.
        channel: channel index, 1 or 2.
        sensorType: sensor type: 0: PT100; 1: PT1000; 2: NTC1; 3: NTC2; 4: Thermo; 5: AD590; 6: EXT1; 7: EXT2.
    Returns: 
        0: Success; negative number: failed.
    """
    return SetSensorType(hdl, channel, sensorType)

def TC300GetSensorParameter(hdl, channel, parameter):
    """  Get sensor parameter when sensor type is PT100
    Args:
         hdl: handle of port.
         channel: channel index, 1 or 2.
         parameter: sensor parameter: 0: 2 wire; 1: 3 wire; 2: 4 wire; 3: J type; 4: K type.
    Returns: 
         0: Success; negative number: failed
    """
    val = c_int(0)
    ret = GetSensorParameter(hdl,channel,val)
    parameter[0] = val.value
    return ret

def TC300SetSensorParameter(hdl, channel, parameter):
    """ Set sensor parameter when sensor type is PT100
    Args:
        hdl: handle of port.
        channel: channel index, 1 or 2.
        parameter: sensor parameter: 0: 2 wire; 1: 3 wire; 2: 4 wire; 3: J type; 4: K type.
    Returns: 
        0: Success; negative number: failed.
    """
    return SetSensorParameter(hdl, channel, parameter)


def TC300GetNtcBeta(hdl, channel, constant):
    """  Get β value when sensor type is ntc1
    args:
         hdl: handle of port.
         channel: channel index, 1 or 2.
         constant: the β value for ntc1 sensor.
    returns: 
         0: success; negative number: failed
    """
    val = c_float(0)
    ret = GetNtcBeta(hdl,channel,val)
    constant[0] = val.value
    return ret

def TC300SetNtcBeta(hdl, channel, constant):
    """ Set β value when sensor type is ntc1
    args:
        hdl: handle of port.
        channel: channel index, 1 or 2.
        constant: the β value for ntc1 sensor 0~9999.
    returns: 
        0: success; negative number: failed.
    """
    return SetNtcBeta(hdl, channel, constant)

def TC300GetExtBeta(hdl, channel, constant):
    """  Get β value when sensor type is EXT1
    args:
         hdl: handle of port.
         channel: channel index, 1 or 2.
         constant: the β value for EXT1 sensor.
    returns: 
         0: success; negative number: failed
    """
    val = c_float(0)
    ret = GetExtBeta(hdl,channel,val)
    constant[0] = val.value
    return ret

def TC300SetExtBeta(hdl, channel, constant):
    """ Set β value when sensor type is EXT1
    args:
        hdl: handle of port.
        channel: channel index, 1 or 2.
        constant: the β value for EXT1 sensor 0~9999.
    returns: 
        0: success; negative number: failed.
    """
    return SetExtBeta(hdl, channel, constant)




def TC300GetTriggerMode(hdl, channel, mode):
    """  Get trigger mode
    Args:
         hdl: handle of port.
         channel: channel index, 1 or 2.
         mode: trigger mode: 0: output; 1: input.
    Returns: 
         0: Success; negative number: failed
    """
    val = c_int(0)
    ret = GetTriggerMode(hdl,channel,val)
    mode[0] = val.value
    return ret

def TC300SetTriggerMode(hdl, channel, mode):
    """ Set trigger mode
    Args:
        hdl: handle of port.
        channel: channel index, 1 or 2.
        mode: trigger mode: 0: output; 1: input.
    Returns: 
        0: Success; negative number: failed.
    """
    return SetTriggerMode(hdl, channel, mode)

def TC300GetBrightness(hdl, brightness):
    """  Get LCD brightness
    Args:
         hdl: handle of port.
         brightness: LCD brightness
    Returns: 
         0: Success; negative number: failed
    """
    val = c_int(0)
    ret = GetBrightness(hdl, val)
    brightness[0] = val.value
    return ret

def TC300SetBrightness(hdl, brightness):
    """ Set LCD brightness
    Args:
        hdl: handle of port.        
        brightness: LCD brightness 10~100.
    Returns: 
        0: Success; negative number: failed.
    """
    return SetBrightness(hdl, brightness)

def TC300GetKnobState(hdl, state):
    """  Get knob state
    Args:
         hdl: handle of port.
         state: knob state: 0: unlocked; 1: locked.
    Returns: 
         0: Success; negative number: failed
    """
    val = c_int(0)
    ret = GetKnobState(hdl,val)
    state[0] = val.value
    return ret

def TC300SetKnobState(hdl, state):
    """ Set knob state
    Args:
        hdl: handle of port.
        state: knob state: 0: unlocked; 1: locked.
    Returns: 
        0: Success; negative number: failed.
    """
    return SetKnobState(hdl, state)

def TC300GetPIDParameter(hdl, channel, p, i, d, period):
    """  Get PID parameters
    Args:
         hdl: handle of port.
         channel: channel index, 1 or 2.
         p: Kp
         i: Ti
         d: Td
         period: period time
    Returns: 
         0: Success; negative number: failed
    """
    kp = c_float(0)
    ti = c_float(0)
    td = c_float(0)
    peri = c_int(0)
    ret = GetPIDParameter(hdl,channel,kp,ti,td,peri)
    p[0] = kp.value
    i[0] = ti.value
    d[0] = td.value
    period[0] = peri.value
    return ret

def TC300GetPIDParameterP(hdl, channel, p):
    """  PID parameter Kp
    Args:
         hdl: handle of port.
         channel: channel index, 1 or 2.
         p: the PID parameter Kp 
    Returns: 
         0: Success; negative number: failed
    """
    val = c_float(0)
    ret = GetPIDParameterP(hdl,channel,val)
    p[0] = val.value
    return ret

def TC300SetPIDParameterP(hdl, channel, p):
    """ Set PID parameter Kp
    Args:
        hdl: handle of port.
        channel: channel index, 1 or 2.
        p: the PID parameter Kp 0~9.99 A/K
    Returns: 
        0: Success; negative number: failed.
    """
    return SetPIDParameterP(hdl, channel, p)

def TC300GetPIDParameterI(hdl, channel, i):
    """  PID parameter Ti
    Args:
         hdl: handle of port.
         channel: channel index, 1 or 2.
         i: the PID parameter Ki 
    Returns: 
         0: Success; negative number: failed
    """
    val = c_float(0)
    ret = GetPIDParameterI(hdl,channel,val)
    i[0] = val.value
    return ret

def TC300SetPIDParameterI(hdl, channel, i):
    """ Set PID parameter Ti
    Args:
        hdl: handle of port.
        channel: channel index, 1 or 2.
        i: the PID parameter Ti 0~9.99 A/(K*sec)
    Returns: 
        0: Success; negative number: failed.
    """
    return SetPIDParameterI(hdl, channel, i)

def TC300GetPIDParameterD(hdl, channel, d):
    """  PID parameter Td
    Args:
         hdl: handle of port.
         channel: channel index, 1 or 2.
         i: the PID parameter Kd 
    Returns: 
         0: Success; negative number: failed
    """
    val = c_float(0)
    ret = GetPIDParameterD(hdl,channel,val)
    d[0] = val.value
    return ret

def TC300SetPIDParameterD(hdl, channel, d):
    """ Set PID parameter Td
    Args:
        hdl: handle of port.
        channel: channel index, 1 or 2.
        d: the PID parameter Td 0~9.99 (A*sec)/K
    Returns: 
        0: Success; negative number: failed.
    """
    return SetPIDParameterD(hdl, channel, d)

def TC300GetPIDParameterPeriod(hdl, channel, period):
    """  Get PID period time
    Args:
         hdl: handle of port.
         channel: channel index, 1 or 2.
         period: the PID period time
    Returns: 
         0: Success; negative number: failed
    """
    val = c_int(0)
    ret = GetPIDParameterPeriod(hdl,channel,val)
    period[0] = val.value
    return ret

def TC300SetPIDParameterPeriod(hdl, channel, period):
    """ Set PID period time
    Args:
        hdl: handle of port.
        channel: channel index, 1 or 2.
        period: the PID period time 100~5000 ms
    Returns: 
        0: Success; negative number: failed.
    """
    return SetPIDParameterPeriod(hdl, channel, period)

def TC300LoadDefaultPIDParameter(hdl, channel):
    """ Reset PID parameters to default
    Args:
        hdl: handle of port.
        channel: channel index, 1 or 2.
    Returns: 
        0: Success; negative number: failed.
    """
    return LoadDefaultPIDParameter(hdl, channel)

def TC300GetMonitorMessage(hdl, ch1TargTemp, ch1Temp, ch1TargCurr, ch1Curr, ch1MinTemp, ch1MaxTemp, ch1MaxCurr, ch1Volt, ch1Mode, ch1Trig, ch1SensType,
	ch2TargTemp, ch2Temp, ch2TargCurr, ch2Curr, ch2MinTemp, ch2MaxTemp, ch2MaxCurr,ch2Volt, ch2Mode, ch2Trig, ch2SensType):
    """  Get all status in one command
    Args:
         hdl: handle of port.
         ch1TargTemp: channel 1 target temperature.
         ch1Temp: channel 1 actual temperature.
         ch1TargCurr: channel 1 target current.
         ch1Curr: channel 1 current.
         ch1MinTemp: channel 1 minimum temperature.
         ch1MaxTemp: channel 1 maximum temperature.
         ch1MaxCurr: channel 1 maximum current.
         ch1Volt: channel 1 voltage.
         ch1Mode: channel 1 mode: 0: Heater; 1: Tec; 2: Constant current;.
         ch1Trig: channel 1 trigger mode: 0: output; 1: input.
         ch1SensType: channel 1 sensor type: 0: PT100; 1: PT1000; 2: NTC10K; 3: Thermo; 4: AD590; 5: Ext TSP-TH.
         ch2TargTemp: channel 2 target temperature.
         ch2Temp: channel 2 actual temperature.
         ch2TargCurr: channel 2 target current.
         ch2Curr: channel 2 current.
         ch2MinTemp: channel 2 minimum temperature.
         ch2MaxTemp: channel 2 maximum temperature.
         ch2MaxCurr: channel 2 maximum current.
         ch2Volt: channel 2 voltage.
         ch2Mode: channel 2 mode: 0: Heater; 1: Tec; 2: Constant current; 3: Synchronize with Ch1.
         ch2Trig: channel 2 trigger mode: 0: output; 1: input.
         ch2SensType: channel 2 sensor type: 0: PT100; 1: PT1000; 2: NTC10K; 3: Thermo; 4: AD590; 5: Ext TSP-TH.
    Returns: 
         0: Success; negative number: failed
    """    
    ch1TargTempval = c_float(0)
    ch1Tempval = c_float(0)
    ch1TargCurrval = c_float(0)
    ch1Currval = c_float(0)
    ch1MinTempval = c_float(0)
    ch1MaxTempval = c_float(0)
    ch1MaxCurrval = c_float(0)
    ch1Voltval = c_float(0)
    ch1Modeval = c_int(0)
    ch1Trigval = c_int(0)
    ch1SensTypeval = c_int(0)
    ch2TargTempval = c_float(0)
    ch2Tempval = c_float(0)
    ch2TargCurrval = c_float(0)
    ch2Currval = c_float(0)
    ch2MinTempval = c_float(0)
    ch2MaxTempval = c_float(0)
    ch2MaxCurrval = c_float(0)
    ch2Voltval = c_float(0)
    ch2Modeval = c_int(0)
    ch2Trigval = c_int(0)
    ch2SensTypeval = c_int(0)
    ret = GetMonitorMessage(hdl, ch1TargTempval, ch1Tempval,ch1TargCurrval, ch1Currval,ch1MinTempval,ch1MaxTempval,ch1MaxCurrval, ch1Voltval, ch1Modeval, ch1Trigval, ch1SensTypeval, 
                            ch2TargTempval, ch2Tempval, ch2TargCurrval, ch2Currval, ch2MinTempval, ch2MaxTempval, ch2MaxCurrval, ch2Voltval, ch2Modeval, ch2Trigval, ch2SensTypeval)
    ch1TargTemp[0] = ch1TargTempval.value
    ch1Temp[0] = ch1Tempval.value
    ch1TargCurr[0]=ch1TargCurrval.value
    ch1Curr[0] = ch1Currval.value
    ch1MinTemp[0]=ch1MinTempval.value
    ch1MaxTemp[0]=ch1MaxTempval.value
    ch1MaxCurr[0]=ch1MaxCurrval.value
    ch1Volt[0] = ch1Voltval.value
    ch1Mode[0] = ch1Modeval.value
    ch1Trig[0] = ch1Trigval.value
    ch1SensType[0] = ch1SensTypeval.value
    ch2TargTemp[0] = ch2TargTempval.value
    ch2Temp[0] = ch2Tempval.value
    ch2TargCurr[0]=ch2TargCurrval.value
    ch2Curr[0] = ch2Currval.value
    ch2MinTemp[0] = ch2MinTempval.value
    ch2MaxTemp[0] = ch2MaxTempval.value
    ch2MaxCurr[0] = ch2MaxCurrval.value
    ch2Volt[0] = ch2Voltval.value
    ch2Mode[0] = ch2Modeval.value
    ch2Trig[0] = ch2Trigval.value
    ch2SensType[0] = ch2SensTypeval.value
    return ret

def TC300GetErrorMessage(hdl, message):
    """  Get error message when GetStatus result contains an error
    Args:
         hdl: handle of port.
         message: 
            Device state define 	8bit
            bit0 = 1	No Load Ch1
            bit1 = 1	No Sensor Ch1
            bit2 = 1	No Load Ch2
            bit3 = 1	No Sensor Ch2
            bit4 = 1	Future definition
            bit5 = 1	Future definition
            bit6 = 1	Future definition
            bit7 = 1	Hsink Temper High
    Returns: 
         0: Success; negative number: failed
    """
    val = c_ubyte(0)
    ret = GetErrorMessage(hdl,val)
    message[0] = val.value
    return ret

def TC300GetWarningMessage(hdl, message):
    """  Get warning message when GetStatus result contains an error
    Args:
         hdl: handle of port.
         message: 
            Device state define 	8bit
            bit0 = 1	Amb Temper High
            bit1 = 1	Ch1 Temper High
            bit2 = 1	Ch1 Temper Low
            bit3 = 1	Ch2 Temper High
            bit4 = 1	Ch2 Temper Low
            bit5 = 1	Future definition
            bit6 = 1	Future definition
             bit7 = 1	Future definition
    Returns: 
         0: Success; negative number: failed
    """
    val = c_ubyte(0)
    ret = GetWarningMessage(hdl,val)
    message[0] = val.value
    return ret

def TC300LoadFactoryParameter(hdl):
    """ Restore factory settings
    Args:
        hdl: handle of port.
    Returns: 
        0: Success; negative number: failed.
    """
    return LoadFactoryParameter(hdl)

#endregion

