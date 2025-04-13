
try:
    from TC300_COMMAND_LIB import *
    import time
except OSError as ex:
    print("Warning:",ex)
         

# ------------ Example Channel Read&Write for channel 1-------------- # 
def ChannelReadWrite(hdl):
   print("*** Channel Read&Write example")   

   #set channel count to 2
   result = TC300SetChannels(hdl, 1)  #  channel number:0: single channel; 1: dual channel
   if(result<0):
       print("Set dual channel fail" , result)
   else:
       print("channel number is dual channel")

   channels=[0]
   channelsList={0:"single channel", 1:"dual channel"}
   result=TC300GetChannels(hdl,channels)
   if(result<0):
      print("Get channel number fail",result)
   else:
      print("Get channel number :",channelsList.get(channels[0]))

   #Enable channel 1
   result = TC300EnableChannel(hdl, 1, 1)  #  0: Disable; 1: Enable
   if(result<0):
       print("Set channel 1 enabled fail" , result)
   else:
       print("channel 1 enabled")

   result = TC300SetMode(hdl, 1, 2) #channel mode: 0: Heater; 1: Tec; 2: Constant current; 3: Synchronize with Ch1(only for channel 2)
   if(result<0):
       print("Set mode fail" , result)
   else:
       print("mode is" , "Constant current")

   mode=[0]
   modeList={0: "Heater", 1: "Tec", 2: "Constant current", 3: "Synchronize with Ch1(only for channel 2)"}
   result=TC300GetMode(hdl,1,mode)
   if(result<0):
      print("Get mode fail",result)
   else:
      print("Get mode :",modeList.get(mode[0]))

   result = TC300SetOutputCurrent(hdl, 1, 500)  #  the output current -2000~2000mA
   if(result<0):
       print("Set target Output Current fail" , result)
   else:
       print("Sensor target Output Current is" , 500)

   OutputCurrent=[0]
   result=TC300GetTargetOutputCurrent(hdl,1,OutputCurrent)
   if(result<0):
      print("Get target Output Current fail",result)
   else:
      print("target Output Current is :",OutputCurrent[0])

   result = TC300SetTargetTemperature(hdl, 1, 50)  #   the target temperature -200~400 °C
   if(result<0):
       print("Set Target Temperature fail" , result)
   else:
       print("Target Temperature is" , 50)

   TargetTemperature=[0]
   result=TC300GetTargetTemperature(hdl,1,TargetTemperature)
   if(result<0):
      print("Get Target Temperature fail",result)
   else:
      print("Target Temperature is :",TargetTemperature[0])

   ActualTemperature=[0]
   result=TC300GetActualTemperature(hdl,1,ActualTemperature)
   if(result<0):
      print("Get Actual Temperature fail",result)
   else:
      print("Actual Temperature is :",ActualTemperature[0])   

   # Set channel 2 mode to Synchronize with Ch1(only for channel 2)
   result = TC300SetMode(hdl, 2, 3) #channel mode: 0: Heater; 1: Tec; 2: Constant current; 3: Synchronize with Ch1(only for channel 2)
   if(result<0):
       print("Set mode fail" , result)
   else:
       print("mode is" , "Synchronize with Ch1(only for channel 2)")

   mode=[0]
   modeList={0: "Heater", 1: "Tec", 2: "Constant current", 3: "Synchronize with Ch1(only for channel 2)"}
   result=TC300GetMode(hdl,2,mode)
   if(result<0):
      print("Get mode fail",result)
   else:
      print("Get mode :",modeList.get(mode[0]))  


# ------------ Example channel Parameters Read&Write for channel 1  -------------- # 
def ChannelParametersReadWrite(hdl):
   print("*** Channel Parameters Read&Write example")

   result = TC300SetTriggerMode(hdl, 1, 0) #0:output,1:input
   if(result<0):
       print("Set Trigger Mode fail" , result)
   else:
       print("Trigger Mode is" , "output")

   triggermode=[0]
   triggermodeList={0:"output", 1:"input"}
   result=TC300GetTriggerMode(hdl, 1, triggermode)
   if(result<0):
      print("Get Trigger Mode fail",result)
   else:
      print("Get Trigger Mode :",triggermodeList.get(triggermode[0]))

   result = TC300SetMinTemperature(hdl, 1, 10)  # the min value of temperature -200~50 °C
   if(result<0):
       print("Set Min Temperature fail" , result)
   else:
       print("Sensor Min Temperature is" , 10)

   MinTemperature=[0]
   result=TC300GetMinTemperature(hdl,1,MinTemperature)
   if(result<0):
      print("Get Min Temperature fail",result)
   else:
      print("Min Temperature is :",MinTemperature[0])

   result = TC300SetMaxTemperature(hdl, 1, 300)  # the max value of temperature 50~400 °C
   if(result<0):
       print("Set Max Temperature fail" , result)
   else:
       print("Sensor Max Temperature is" , 300)

   MaxTemperature=[0]
   result=TC300GetMaxTemperature(hdl,1,MaxTemperature)
   if(result<0):
      print("Get Max Temperature fail",result)
   else:
      print("Max Temperature is :",MaxTemperature[0])

   result = TC300SetMaxCurrent(hdl, 1, 1500)  # the max value of current 0~2000 mA
   if(result<0):
       print("Set Max Current fail" , result)
   else:
       print("Sensor Max Current is" , 1500)

   MaxCurrent=[0]
   result=TC300GetMaxCurrent(hdl,1,MaxCurrent)
   if(result<0):
      print("Get Max Current fail",result)
   else:
      print("Max Current is :",MaxCurrent[0])

   result = TC300SetMaxVoltage(hdl, 1, 12.3)  #  the max value of voltage 0.1-24.0 V
   if(result<0):
       print("Set Max Voltage fail" , result)
   else:
       print("Max Voltage is" , 12.3)

   MaxVoltage=[0]
   result=TC300GetMaxVoltage(hdl,1,MaxVoltage)
   if(result<0):
      print("Get Max Voltage fail",result)
   else:
      print("Max Voltage is :",MaxVoltage[0])

   result = TC300SetQuietMode(hdl, 1) #0: disable; 1: enable
   if(result<0):
       print("Set quiet mode fail" , result)
   else:
       print("quiet mode is enable")

   mode=[0]
   modeList={0:"disable", 1:"enable"}
   result=TC300GetQuietMode(hdl, mode)
   if(result<0):
      print("Get quiet mode fail",result)
   else:
      print("Get quiet mode :",modeList.get(mode[0]))

   result = TC300SetDarkStatus(hdl, 0) #0: disable; 1: enable
   if(result<0):
       print("Set dark status fail" , result)
   else:
       print("dark status is disable")

   status=[0]
   statusList={0:"disable", 1:"enable"}
   result=TC300GetDarkStatus(hdl, status)
   if(result<0):
      print("Get dark status fail",result)
   else:
      print("Get dark status :",statusList.get(status[0]))           



# ------------ Example Sensor Type Parameters Read&Write for channel 1  -------------- # 
def SensorTypeParametersReadWrite(hdl):
   print("*** Sensor Type Parameters Read&Write example")

   #As Constant current does not need sensor, so Channel mode should be Heater or Tec
   result = TC300SetMode(hdl, 1, 0) #channel mode: 0: Heater; 1: Tec; 2: Constant current; 3: Synchronize with Ch1(only for channel 2)
   if(result<0):
       print("Set mode fail" , result)
   else:
       print("mode is" , "Heater")

   SensorT_str = input("Please set the sensor type here(just input the number): (0: PT100; 1: PT1000; 2: NTC1; 3: NTC2; 4: Thermo; 5: AD590; 6: EXT1; 7: EXT2)")
   SensorT = int(SensorT_str)

   result = TC300SetSensorType(hdl, 1, SensorT) # 0: PT100; 1: PT1000; 2: NTC1; 3: NTC2; 4: Thermo; 5: AD590; 6: EXT1; 7: EXT2
   if(result<0):
       print("Set Sensor Type fail" , result)
   else:
       print("Set Sensor Type successfully")

   SensorType=[0]
   SensorTypeList={0: "PT100", 1: "PT1000", 2: "NTC1", 3: "NTC2", 4: "Thermo", 5: "AD590", 6:"EXT1", 7:"EXT2"}
   result=TC300GetSensorType(hdl,1,SensorType)
   if(result<0):
      print("Get Sensor Type fail",result)
   else:
      print("Get Sensor Type :",SensorTypeList.get(SensorType[0]))

   #PT100/PT1000 parameter settings(sensor parameter(2 wire or 4 wire) and sensor offset)  
   if(SensorT == 0 or SensorT == 1): 
      result = TC300SetSensorParameter(hdl, 1, 0) #0: 2 wire; 1: 3 wire; 2: 4 wire; 3: J type; 4: K type
      if(result<0):
          print("Set Sensor Parameter fail" , result)
      else:
          print("Sensor Parameter is" , " 2 wire")

      SensorParameter=[0]
      SensorParameterList={0: "2 wire", 1: "3 wire", 2: "4 wire", 3: "J type", 4: "K type"}
      result=TC300GetSensorParameter(hdl,1,SensorParameter)
      if(result<0):
         print("Get Sensor Parameter fail",result)
      else:
         print("Sensor Parameter is :",SensorParameterList.get(SensorParameter[0]))

      result = TC300SetSensorOffset(hdl, 1, 5) #the offset value for PT100/PT1000, range: -10-10°C.
      if(result<0):
          print("Set Sensor offset fail" , result)
      else:
          print("Sensor offset is" , 5)

      SensorOffset=[0]
      result=TC300GetSensorOffset(hdl,1,SensorOffset)
      if(result<0):
         print("Get Sensor offset fail",result)
      else:
         print("Sensor offset is :",SensorOffset[0]) 

   #Thermo parameter settings(sensor parameter: J type or K type)
   elif(SensorT == 4):                       
      result = TC300SetSensorParameter(hdl, 1, 3) #0: 2 wire; 1: 3 wire; 2: 4 wire; 3: J type; 4: K type
      if(result<0):
         print("Set Sensor Parameter fail" , result)
      else:
         print("Sensor Parameter is" , " J type")

      SensorParameter=[0]
      SensorParameterList={0: "2 wire", 1: "3 wire", 2: "4 wire", 3: "J type", 4: "K type"}
      result=TC300GetSensorParameter(hdl,1,SensorParameter)
      if(result<0):
         print("Get Sensor Parameter fail",result)
      else:
         print("Sensor Parameter is :",SensorParameterList.get(SensorParameter[0]))

   #NTC1 parameter settings(sensor constant,T0 and R0)
   elif(SensorT == 2):                                               
      result = TC300SetNtcBeta(hdl, 1, 1000) #the β value for NTC1 sensor 0~9999.
      if(result<0):
          print("Set NTC Beta fail" , result)
      else:
          print("NTC Beta is" , 1000)

      NtcBeta=[0]
      result=TC300GetNtcBeta(hdl,1,NtcBeta)
      if(result<0):
         print("Get Sensor Constant fail",result)
      else:
         print("Sensor Constant is :",NtcBeta[0])  
  
      result = TC300SetT0Constant(hdl, 1, 100)  # the T0 value when sensor type is NTC1,0-999
      if(result<0):
          print("Set T0 Constant fail" , result)
      else:
          print("Sensor T0 Constant is" , 100)

      T0Constant=[0]
      result=TC300GetT0Constant(hdl,1,T0Constant)
      if(result<0):
         print("Get T0 Constant fail",result)
      else:
         print("Min T0 Constant is :",T0Constant[0])

      result = TC300SetR0Constant(hdl, 1, 200)  # the R0 value when sensor type is NTC1,0-999
      if(result<0):
          print("Set R0 Constant fail" , result)
      else:
          print("Sensor R0 Constant is" , 200)

      R0Constant=[0]
      result=TC300GetR0Constant(hdl,1,R0Constant)
      if(result<0):
         print("Get R0 Constant fail",result)
      else:
         print("Min R0 Constant is :",R0Constant[0])

   #EXT1 parameter settings(sensor constant,T0 and R0)
   elif(SensorT == 6):                                           
   
      result = TC300SetExtBeta(hdl, 1, 2000) #the β value for EXT1 sensor 0~9999.
      if(result<0):
          print("Set EXT1 Beta fail" , result)
      else:
          print("EXT1 Beta is" , 2000)

      EXT1Beta=[0]
      result=TC300GetExtBeta(hdl,1,EXT1Beta)
      if(result<0):
         print("Get Ext1 Beta fail",result)
      else:
         print("Ext1 Beta :",EXT1Beta[0])  

  
      result = TC300SetT0Constant(hdl, 1, 100)  # the T0 value when sensor type is EXT1,0-999
      if(result<0):
          print("Set T0 Constant fail" , result)
      else:
          print("Sensor T0 Constant is" , 100)

      T0Constant=[0]
      result=TC300GetT0Constant(hdl,1,T0Constant)
      if(result<0):
         print("Get T0 Constant fail",result)
      else:
         print("Min T0 Constant is :",T0Constant[0])

      result = TC300SetR0Constant(hdl, 1, 200)  # the R0 value when sensor type is EXT1,0-999
      if(result<0):
          print("Set R0 Constant fail" , result)
      else:
          print("Sensor R0 Constant is" , 200)

      R0Constant=[0]
      result=TC300GetR0Constant(hdl,1,R0Constant)
      if(result<0):
         print("Get R0 Constant fail",result)
      else:
         print("Min R0 Constant is :",R0Constant[0])

   #NTC2/EXT2 parameter settings(Hart A,Hart B and Hart C)
   elif(SensorT == 3 or SensorT == 7):                
      result = TC300SetHartAConstant(hdl, 1, -1.5555) #the Hart A value for NTC2 or EXT2 sensor -9.9999~9.9999
      if(result<0):
          print("Set Hart A value fail" , result)
      else:
          print("Hart A value is" , -1.5555)

      HartA=[0]
      result=TC300GetHartAConstant(hdl,1,HartA)
      if(result<0):
         print("Get Hart A value fail",result)
      else:
         print("Hart A value is :",HartA[0])  
  
      result = TC300SetHartBConstant(hdl, 1, -2.5555) #the Hart B value for NTC2 or EXT2 sensor -9.9999~9.9999
      if(result<0):
         print("Set Hart B value fail" , result)
      else:
         print("Hart B value is" , -2.5555)

      HartB=[0]
      result=TC300GetHartBConstant(hdl,1,HartB)
      if(result<0):
         print("Get Hart B value fail",result)
      else:
         print("Hart B value is :",HartB[0])

      result = TC300SetHartCConstant(hdl, 1, 3.5555) #the Hart C value for NTC2 or EXT2 sensor -9.9999~9.9999
      if(result<0):
          print("Set Hart C value fail" , result)
      else:
          print("Hart C value is" , 3.5555)

      HartC=[0]
      result=TC300GetHartCConstant(hdl,1,HartC)
      if(result<0):
         print("Get Hart C value fail",result)
      else:
         print("Hart C value is :",HartC[0])
    
   elif(SensorT == 5):
       print("No parameter need to be set when sensor type is AD590")
   else:
       print("Wrong sensor number")



      # ------------ Example Channel PID Read&Write demo for channel 1-------------- # 
def ChannelPIDReadWrite(hdl):
   print("*** Channel PID Read&Write example")  

   result = TC300SetPIDParameterD(hdl, 1, 6.66)  #  the PID parameter Td 0~9.99 (A*sec)/K
   if(result<0):
       print("Set Td fail" , result)
   else:
       print("Td is" ,6.66)

   Td=[0]
   result=TC300GetPIDParameterD(hdl,1,Td)
   if(result<0):
      print("Get Td fail",result)
   else:
      print("Td is :",Td[0])

   result = TC300SetPIDParameterI(hdl, 1, 7.77)  #  the PID parameter Ti 0~9.99 A/(K*sec)
   if(result<0):
       print("Set Ti fail" , result)
   else:
       print("Ti is" , 7.77)

   Ti=[0]
   result=TC300GetPIDParameterI(hdl,1,Ti)
   if(result<0):
      print("Get Ti fail",result)
   else:
      print("Ti is :",Ti[0])

   result = TC300SetPIDParameterP(hdl, 1, 8.88)  #  the PID parameter Kp 0~9.99 A/K
   if(result<0):
       print("Set Kp fail" , result)
   else:
       print("Kp is" , 8.88)

   Kp=[0]
   result=TC300GetPIDParameterP(hdl,1,Kp)
   if(result<0):
      print("Get Kp fail",result)
   else:
      print("Kp is :",Kp[0])

   result = TC300SetPIDParameterPeriod(hdl, 1, 4000)  #  the PID period time 100~5000 ms
   if(result<0):
       print("Set PID Parameter Period fail" , result)
   else:
       print("PID Parameter Period is" , 4000)

   Period=[0]
   result=TC300GetPIDParameterPeriod(hdl,1,Period)
   if(result<0):
      print("Get PID Parameter Period fail",result)
   else:
      print("PID Parameter Period is :",Period[0])   


def main():
    print("*** TC300 device python example ***")
    try:
        devs = TC300ListDevices()
        print(devs)
        if(len(devs)<=0):
           print('There is no devices connected')
           exit()     

        TC300= devs[0]
        serialNumber= TC300[0]
        hdl = TC300Open(serialNumber,115200,3)       
        if(hdl < 0):
          print("Connect ",serialNumber, "failed" )
          return -1;
        else:
          print("Connect ",serialNumber, "successfully")
   
        result = TC300IsOpen(serialNumber)
        if(result<0):
          print("Open failed ")
        else:
          print("TC300 Is Open ")
        
        ChannelReadWrite(hdl)
        print("--------------------------- Channel 1 Read&Write finished-------------------------")
        ChannelParametersReadWrite(hdl)
        print("--------------------------- Channel 1 Parameter Read&Write finished-------------------------")
        SensorTypeParametersReadWrite(hdl)
        print("--------------------------- channel 1 Sensor Type Parameters Read&Write finished-------------------------")        
        ChannelPIDReadWrite(hdl)
        print("--------------------------- Channel 1 PID Read&Write finished-------------------------")
        
        TC300Close(hdl)    
        
               
    except Exception as ex:
        print("Warning:", ex)
    print("*** End ***")
    input()
main()


