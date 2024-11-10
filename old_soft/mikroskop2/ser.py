import serial

ser = serial.Serial('COM6', timeout = 5)
print(ser)

ser.write(b'STATUS\n')

print('ser')
line = ser.readline()
print(line)