import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd
import mplcursors
from scipy.stats import chi2, multivariate_normal

# Define lists to store results
r = []
el = []
az = []

class CVFilter:
    def __init__(self):
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.Sp = np.zeros((6, 1))  # Predicted state vector
        self.Pp = np.eye(6)  # Predicted state covariance matrix
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.prev_Time = 0
        self.Q = np.eye(6)
        self.Phi = np.eye(6)
        self.Z = np.zeros((3, 1)) 
        self.Z1 = np.zeros((3, 1)) # Measurement vector
        self.Z2 = np.zeros((3, 1)) 
        self.first_rep_flag = False
        self.second_rep_flag = False
        self.gate_threshold = 9.21  # 95% confidence interval for Chi-square distribution with 3 degrees of freedom

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        if not self.first_rep_flag:
            self.Z1 = np.array([[x],[y],[z]])
            self.Sf[0] = x
            self.Sf[1] = y
            self.Sf[2] = z
            self.Meas_Time = time
            self.prev_Time = self.Meas_Time
            self.first_rep_flag = True
        elif self.first_rep_flag and not self.second_rep_flag:
            self.Z2 = np.array([[x],[y],[z]])
            self.prev_Time = self.Meas_Time
            self.Meas_Time = time
            dt = self.Meas_Time - self.prev_Time
            self.vx = (self.Z1[0] - self.Z2[0]) / dt
            self.vy = (self.Z1[1] - self.Z2[1]) / dt
            self.vz = (self.Z1[2] - self.Z2[2]) / dt

            self.Meas_Time = time
            self.second_rep_flag = True
        else:
            self.Z = np.array([[x],[y],[z]])
            self.prev_Time = self.Meas_Time
            self.Meas_Time = time

    def predict_step(self, current_time):
        dt = current_time - self.prev_Time
        T_2 = (dt * dt) / 2.0
        T_3 = (dt * dt * dt) / 3.0
        self.Phi[0, 3] = dt
        self.Phi[1, 4] = dt
        self.Phi[2, 5] = dt
              
        self.Q[0, 0] = T_3
        self.Q[1, 1] = T_3
        self.Q[2, 2] = T_3
        self.Q[0, 3] = T_2
        self.Q[1, 4] = T_2
        self.Q[2, 5] = T_2
        self.Q[3, 0] = T_2
        self.Q[4, 1] = T_2
        self.Q[5, 2] = T_2
        self.Q[3, 3] = dt
        self.Q[4, 4] = dt
        self.Q[5, 5] = dt
        self.Q = self.Q * self.plant_noise
        self.Sp = np.dot(self.Phi, self.Sf)
        self.Pp = np.dot(np.dot(self.Phi, self.Pf), self.Phi.T) + self.Q
        self.Meas_Time = current_time

    def update_step(self, Z):
        Inn = Z - np.dot(self.H, self.Sp)
        S = np.dot(self.H, np.dot(self.Pp, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pp, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sp + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pp)

    def gating(self, Z):
        Inn = Z - np.dot(self.H, self.Sp)
        S = np.dot(self.H, np.dot(self.Pp, self.H.T)) + self.R
        d2 = np.dot(np.dot(np.transpose(Inn), np.linalg.inv(S)), Inn)
        return d2 < self.gate_threshold

# Define the CAFilter and CTFilter classes similarly to CVFilter
# Placeholder example for CAFilter and CTFilter
class CAFilter(CVFilter):
    def __init__(self):
        super().__init__()
        # Custom initialization for CAFilter

class CTFilter(CVFilter):
    def __init__(self):
        super().__init__()
        # Custom initialization for CTFilter

class IMMFilter:
    def __init__(self):
        self.cv_filter = CVFilter()
        self.ca_filter = CAFilter()
        self.ct_filter = CTFilter()
        # Additional initialization for IMMFilter

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        self.cv_filter.initialize_filter_state(x, y, z, vx, vy, vz, time)
        self.ca_filter.initialize_filter_state(x, y, z, vx, vy, vz, time)
        self.ct_filter.initialize_filter_state(x, y, z, vx, vy, vz, time)
    
    def predict_step(self, current_time):
        self.cv_filter.predict_step(current_time)
        self.ca_filter.predict_step(current_time)
        self.ct_filter.predict_step(current_time)
        # Combine the predictions from individual filters

    def update_step(self, Z):
        self.cv_filter.update_step(Z)
        self.ca_filter.update_step(Z)
        self.ct_filter.update_step(Z)
        # Combine the updates from individual filters

    def gating(self, Z):
        return self.cv_filter.gating(Z) and self.ca_filter.gating(Z) and self.ct_filter.gating(Z)
    
def form_measurement_groups(measurements, max_time_diff=0.050):
    measurement_groups = []
    current_group = []
    base_time = measurements[0][3]

    for measurement in measurements:
        if measurement[3] - base_time <= max_time_diff:
            current_group.append(measurement)
        else:
            measurement_groups.append(current_group)
            current_group = [measurement]
            base_time = measurement[3]

    if current_group:
        measurement_groups.append(current_group)

    return measurement_groups

def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            mr = float(row[7])  # MR column
            ma = float(row[8])  # MA column
            me = float(row[9])  # ME column
            mt = float(row[10])  # MT column
            x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            r, az, el = cart2sph(x, y, z)  # Convert Cartesian to spherical coordinates
            measurements.append((r, az, el, mt))
    return measurements

def chi_square_clustering(Z, kalman_filter):
    Inn = Z - np.dot(kalman_filter.H, kalman_filter.Sp)
    S = np.dot(kalman_filter.H, np.dot(kalman_filter.Pp, kalman_filter.H.T)) + kalman_filter.R
    d2 = np.dot(np.dot(np.transpose(Inn), np.linalg.inv(S)), Inn)
    gate_threshold = kalman_filter.gate_threshold
    print("gate thres:", gate_threshold)
    print("d2", d2)
    return d2 < gate_threshold

def form_clusters(measurements, kalman_filter):
    clusters = []
    for measurement in measurements:
        Z = np.array([[measurement[0]], [measurement[1]], [measurement[2]]])
        if chi_square_clustering(Z, kalman_filter):
            clusters.append(measurement)
    return clusters

def generate_hypotheses(clusters):
    hypotheses = []
    for cluster in clusters:
        hypotheses.append(cluster)
    return hypotheses

def compute_hypothesis_likelihood(hypothesis, kalman_filter):
    Z = np.array([[hypothesis[0]], [hypothesis[1]], [hypothesis[2]]])
    Inn = Z - np.dot(kalman_filter.H, kalman_filter.Sp)
    S = np.dot(kalman_filter.H, np.dot(kalman_filter.Pp, kalman_filter.H.T)) + kalman_filter.R
    likelihood = multivariate_normal.pdf(Inn.flatten(), mean=np.zeros(3), cov=S)
    return likelihood

def select_best_hypothesis(hypotheses, kalman_filter):
    best_hypothesis = None
    max_likelihood = -float('inf')
    for hypothesis in hypotheses:
        likelihood = compute_hypothesis_likelihood(hypothesis, kalman_filter)
        if likelihood > max_likelihood:
            max_likelihood = likelihood
            best_hypothesis = hypothesis
    return best_hypothesis

def main():
    imm_filter = IMMFilter()

    file_path = "path_to_your_csv.csv"
    measurements = read_measurements_from_csv(file_path)
    measurement_groups = form_measurement_groups(measurements)

    for group in measurement_groups:
        for measurement in group:
            x, y, z, vx, vy, vz, time = measurement
            imm_filter.initialize_filter_state(x, y, z, vx, vy, vz, time)
            imm_filter.predict_step(time)
        
        clusters = form_clusters(group, imm_filter.cv_filter)
        hypotheses = generate_hypotheses(clusters)
        best_hypothesis = select_best_hypothesis(hypotheses, imm_filter.cv_filter)
        
        if best_hypothesis is not None:
            Z = np.array([[best_hypothesis[0]], [best_hypothesis[1]], [best_hypothesis[2]]])
            imm_filter.update_step(Z)

    r.append(imm_filter.cv_filter.Sf[0])
    az.append(imm_filter.cv_filter.Sf[1])
    el.append(imm_filter.cv_filter.Sf[2])

    plt.figure()
    plt.plot(r, label='Range')
    plt.plot(az, label='Azimuth')
    plt.plot(el, label='Elevation')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
