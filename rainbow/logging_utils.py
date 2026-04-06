# -*- coding: utf-8 -*-
"""
Efficient logging utilities for Rainbow DQN training
"""
import json
import os
from datetime import datetime
import threading


class EpisodeLogger:
    """Thread-safe episode logger with efficient file operations"""

    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.episode_file = os.path.join(results_dir, 'episodes.json')
        self.temp_file = os.path.join(results_dir, 'episodes_temp.json')
        self.lock = threading.Lock()

        if not os.path.exists(self.episode_file):
            with open(self.episode_file, 'w') as f:
                json.dump([], f)

    def log_episode(self, episode_info):
        with self.lock:
            try:
                with open(self.episode_file, 'r') as f:
                    data = json.load(f)

                data.append(episode_info)

                with open(self.temp_file, 'w') as f:
                    json.dump(data, f, indent=2)

                os.rename(self.temp_file, self.episode_file)

            except Exception as e:
                print(f"Error logging episode: {e}")
                with open(os.path.join(self.results_dir, 'episodes_fallback.log'), 'a') as f:
                    f.write(json.dumps(episode_info) + '\n')


class EvaluationLogger:
    """Efficient evaluation logger"""

    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.lock = threading.Lock()

    def log_evaluation(self, evaluation_step, evaluation_data):
        with self.lock:
            try:
                eval_filename = f'evaluation_T_{evaluation_step}.json'
                eval_filepath = os.path.join(self.results_dir, eval_filename)
                temp_filepath = os.path.join(self.results_dir, f'evaluation_T_{evaluation_step}_temp.json')

                with open(temp_filepath, 'w') as f:
                    json.dump(evaluation_data, f, indent=2)

                os.rename(temp_filepath, eval_filepath)

            except Exception as e:
                print(f"Error logging evaluation: {e}")
                with open(os.path.join(self.results_dir, 'evaluation_fallback.log'), 'a') as f:
                    f.write(f"T_{evaluation_step}: {json.dumps(evaluation_data)}\n")


class StreamingLogger:
    """Streaming logger for real-time monitoring"""

    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.episode_stream = open(os.path.join(results_dir, 'episodes_stream.jsonl'), 'a')
        self.eval_stream = open(os.path.join(results_dir, 'evaluation_stream.jsonl'), 'a')
        self.lock = threading.Lock()

    def log_episode(self, episode_info):
        with self.lock:
            try:
                self.episode_stream.write(json.dumps(episode_info) + '\n')
                self.episode_stream.flush()
            except Exception as e:
                print(f"Error streaming episode: {e}")

    def log_evaluation_episode(self, eval_info):
        with self.lock:
            try:
                self.eval_stream.write(json.dumps(eval_info) + '\n')
                self.eval_stream.flush()
            except Exception as e:
                print(f"Error streaming evaluation: {e}")

    def close(self):
        self.episode_stream.close()
        self.eval_stream.close()


def create_loggers(results_dir):
    os.makedirs(results_dir, exist_ok=True)

    episode_logger = EpisodeLogger(results_dir)
    evaluation_logger = EvaluationLogger(results_dir)
    streaming_logger = StreamingLogger(results_dir)

    return episode_logger, evaluation_logger, streaming_logger
