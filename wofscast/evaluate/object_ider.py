# For the object identification, matching, and verification
import sys
sys.path.insert(0, '/home/monte.flora/python_packages/MontePython')
import monte_python

import numpy as np 


class ObjectIder:
    def __init__(self, qc_params=None):
        if qc_params is None:
            qc_params = [('min_area', 12)]
        self.qc_params = qc_params
        
        self.qcer = monte_python.QualityControler()

    def label_single_time_step(self, dataset, variable, time_index, 
                               method='single_threshold', params={'bdry_thresh': 0}):
        """
        Labels a single time step of the dataset.
        """
        # Extract the data for a single time step
        input_data = dataset[variable].isel(time=time_index)

        # Apply the labeling method
        labels, object_props = monte_python.label(
            input_data=input_data,
            method=method,
            return_object_properties=True,
            params=params
        )

        # Apply quality control
        labels_qc, object_props_qc = self.qcer.quality_control(
            input_data, labels, object_props, self.qc_params
        )

        return labels_qc, object_props_qc

    def label(self, dataset, variable, method='single_threshold', params={'bdry_thresh': 0}):
        """
        Apply the labeling function for each time step in the dataset.
        """
        # Get the number of time steps
        num_times = dataset.sizes['time']

        # Use Dask to apply labeling for each time step in parallel
        def process_time_step(time_index):
            return self.label_single_time_step(dataset, variable, time_index, method, params)

        # Initialize empty arrays to store the labels
        labels = np.zeros((num_times, dataset.dims['lat'], dataset.dims['lon']))

        # Iterate over each time step and apply labeling
        for time_index in range(num_times):
            labels[time_index, ...] = process_time_step(time_index)[0]
        
        # Assign the dynamically generated labels name (e.g., "COMPOSITE_REFL_10CM_labels")
        dataset = dataset.assign(storms= (('time', 'lat', 'lon'), labels))


        return dataset