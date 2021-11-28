class TimingPerformance(object):
    def __init__(self, exp_label, qpu_response):
        self.exp_label = exp_label
        self.time_labels = [
                            'qpu_sampling_time',
                            'qpu_anneal_time_per_sample',
                            'qpu_readout_time_per_sample',
                            'qpu_access_time',
                            'qpu_access_overhead_time',
                            'qpu_programming_time',
                            'qpu_delay_time_per_sample',
                            'total_post_processing_time',
                            'post_processing_overhead_time'
                            ]
        self.total_time = 0
        self.time_dict = {}

        for t in self.time_labels:
            time = qpu_response.info['timing'][t]
            self.time_dict[t] = time
            self.total_time += time

    def get_time(self, time_label):
        return self.time_dict[time_label]
