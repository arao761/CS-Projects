import serial
import numpy as np
import time 

model = load_model('processed_data.h5')

arduino = serial.Serial('COM3', 9600)  
time.sleep(2) 

def read_emg_data():
    data = arduino.readline().decode('utf-8').rstrip()
    if data:    
        sensor_values = data.split(';')[0]  
        sensor_array = np.array([float(val) for val in sensor_values.split(',') if val])  
        return sensor_array
    return None

def preprocess_data(emg_data, num_sensors, time_steps):
    processed_data = emg_data.reshape(1, time_steps, num_sensors, 1)
    return processed_data
    
def predict_movement(processed_data):
    prediction = model.predict(processed_data)
    return 'OPEN_FINGER' if np.argmax(prediction) == 0 else 'CLOSE_FINGER'

def send_command_to_arduino(command):
    print(f"Sending command to Arduino: {command}")
    arduino.write(command.encode())
    time.sleep(0.1) 

try:
    arduino.flushInput()
    while True:
        emg_data = read_emg_data()
        if emg_data is not None:
            processed_data = preprocess_data(emg_data, num_sensors=3, time_steps=1)  
            command = predict_movement(processed_data)
            send_command_to_arduino(command)
except KeyboardInterrupt:
    arduino.close()
finally:
    arduino.close()