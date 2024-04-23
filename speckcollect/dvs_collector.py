import samna, samnagui
import time
import os
import multiprocessing
import threading
import numpy as np
import json
import sys

class Collector:
    def __init__(self, data_dir, exp_name, time_int=1.0):
        super().__init__()
        self.streamer_endpoint = "tcp://0.0.0.0:40000"
        self.time_int = time_int
        self.data_dir = data_dir
        self.exp_name = exp_name
        os.makedirs(os.path.join(self.data_dir, self.exp_name), exist_ok=True)
        self.debug("Collector initialized with data directory: {} and experiment name: {}".format(self.data_dir, self.exp_name))

    def debug(self, message):
        print(message, file=sys.stdout)
        sys.stdout.flush()

    def open_speck2f_dev_kit(self):
        devices = [
            device
            for device in samna.device.get_unopened_devices()
            if device.device_type_name.startswith("Speck2f")
        ]
        self.debug("Devices found: {}".format(len(devices)))
        assert devices, "Speck2f board not found"

        self.default_config = samna.speck2fBoards.DevKitDefaultConfig()
        self.debug("Opening Speck2f device with default configuration.")
        return samna.device.open_device(devices[0], self.default_config)

    def build_samna_event_route(self, graph, dk):
        _, _, streamer = graph.sequential(
            [dk.get_model_source_node(), "Speck2fDvsToVizConverter", "VizEventStreamer"]
        )
        self.debug("Graph built for event streaming.")

        config_source, _ = graph.sequential([samna.BasicSourceNode_ui_event(), streamer])

        streamer.set_streamer_endpoint(self.streamer_endpoint)
        self.debug("Streamer endpoint set to {}".format(self.streamer_endpoint))
        if streamer.wait_for_receiver_count() == 0:
            raise Exception('Connecting to visualizer on {} failed'.format(self.streamer_endpoint))

        return config_source

    def open_visualizer(self, window_width, window_height, receiver_endpoint):
        gui_process = multiprocessing.Process(
            target=samnagui.run_visualizer,
            args=(receiver_endpoint, window_width, window_height),
        )
        gui_process.start()
        self.debug("Visualizer process started.")
        return gui_process

    def start_visualizer(self):
        def event_collector():
            file_counter = 0
            while gui_process.is_alive():
                events = sink.get_events()
                timestamp = time.time()
                if events:
                    directory = os.path.join(self.data_dir, self.exp_name)
                    os.makedirs(directory, exist_ok=True)

                    filename = f'events_{file_counter}.npy'
                    file_path = os.path.join(directory, filename)
                    np.save(file_path, np.array(events))
                    file_counter += 1

                    self.debug(f'Processed {timestamp} with {len(events)} events, saved to {filename}')
                time.sleep(self.time_int)

        gui_process = self.open_visualizer(800, 600, self.streamer_endpoint)
        dk = self.open_speck2f_dev_kit()
        graph = samna.graph.EventFilterGraph()
        config_source = self.build_samna_event_route(graph, dk)
        sink = samna.graph.sink_from(dk.get_model().get_source_node())
        visualizer_config = samna.ui.VisualizerConfiguration(
            plots=[samna.ui.ActivityPlotConfiguration(128, 128, "DVS Layer", [0, 0, 1, 1])]
        )
        config_source.write([visualizer_config])
        config = samna.speck2f.configuration.SpeckConfiguration()
        config.dvs_layer.monitor_enable = True
        dk.get_model().apply_configuration(config)
        collector_thread = threading.Thread(target=event_collector)
        collector_thread.start()
        self.debug("Event collector thread started.")
        graph.start()
        gui_process.join()
        graph.stop()
        collector_thread.join()
        self.debug("Event collection stopped.")

    def save_events(self):
        np.save(os.path.join(self.data_dir, self.exp_name + '.npy'), self.event_dict)
        self.debug("Events saved to {}.npy".format(os.path.join(self.data_dir, self.exp_name)))
