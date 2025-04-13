import os

START_POS = 0
END_POS = 399
PATH = "patterns/illumination_files/"
POWER = 60
STANDARD_WAIT = 10
MEASUREMENT_COUNT = 3

load_img_line = "d_load_img('{path}illumination_field_{num}.png')"
power_line = "d_laser_duty({power})"
wait_line = "d_wait({wait_time})"
save_line = "d_ahk(ahk_save_thermal.ahk, '{num}_{power}-{mst}')"



def add_measurement(path, num, power, wait_time, mst = 0):
    lines = [
        load_img_line.format(path=path, num=str(num).zfill(3)),
        power_line.format(power=power),
        wait_line.format(wait_time=wait_time),
        save_line.format(num=str(num).zfill(3), power=power, mst=mst)
    ]
    return lines
    
    
if __name__ == "__main__":
    lines = []
    lines.append("d_laser_switch(1)")
    
    for i in range(START_POS, END_POS):
        for n in range(1, MEASUREMENT_COUNT+1):
            measurement = add_measurement(PATH, i, POWER, STANDARD_WAIT, mst = STANDARD_WAIT*n)
            lines.extend(measurement)
    
    
    lines.append("d_laser_switch(0)")
    
    with open("current_ml_script.txt", "w") as f:
        for line in lines:
            f.write(f"{line}\n")
            print(line)
        
    