#!/usr/bin/env python3
"""
Run parallel processing with real-time process monitoring.
Tracks CPU, memory, and thread activity at the OS level.
"""

import argparse
import json
import os
import psutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


class ProcessMonitor:
    """Monitor process-level CPU, memory, and thread activity."""

    def __init__(self, interval=0.5):
        self.interval = interval
        self.monitoring = False
        self.data = []
        self.start_time = None
        self.process = psutil.Process()

    def start(self):
        """Start monitoring in background thread."""
        self.monitoring = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print(f"\n{'='*80}")
        print("PROCESS MONITORING STARTED")
        print(f"{'='*80}")
        print(f"PID: {self.process.pid}")
        print(f"Monitoring interval: {self.interval}s")
        print(f"{'='*80}\n")

    def stop(self):
        """Stop monitoring and return collected data."""
        self.monitoring = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2)
        return self.data

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                timestamp = time.time() - self.start_time

                # Get process metrics
                cpu_percent = self.process.cpu_percent(interval=0.1)
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                num_threads = self.process.num_threads()

                # Get children processes (for parallel workers)
                children = self.process.children(recursive=True)
                child_cpu = sum(c.cpu_percent(interval=0.1) for c in children)
                child_memory = sum(c.memory_info().rss for c in children) / 1024 / 1024

                # Get system-wide info
                system_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
                system_memory = psutil.virtual_memory()

                snapshot = {
                    'timestamp': timestamp,
                    'process': {
                        'pid': self.process.pid,
                        'cpu_percent': cpu_percent,
                        'memory_mb': memory_mb,
                        'num_threads': num_threads
                    },
                    'children': {
                        'count': len(children),
                        'cpu_percent': child_cpu,
                        'memory_mb': child_memory,
                        'pids': [c.pid for c in children]
                    },
                    'system': {
                        'cpu_percent_per_core': system_cpu,
                        'cpu_percent_total': sum(system_cpu) / len(system_cpu),
                        'memory_used_mb': system_memory.used / 1024 / 1024,
                        'memory_percent': system_memory.percent
                    }
                }

                self.data.append(snapshot)

                # Print live update every 2 seconds
                if len(self.data) % 4 == 0:
                    self._print_status(snapshot)

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

            time.sleep(self.interval)

    def _print_status(self, snapshot):
        """Print current monitoring status."""
        p = snapshot['process']
        c = snapshot['children']
        s = snapshot['system']

        print(f"[{snapshot['timestamp']:6.1f}s] "
              f"CPU: {p['cpu_percent']:5.1f}% | "
              f"Memory: {p['memory_mb']:6.1f}MB | "
              f"Threads: {p['num_threads']:2d} | "
              f"Children: {c['count']} ({c['cpu_percent']:5.1f}% CPU)")

    def generate_report(self, output_path):
        """Generate detailed monitoring report."""
        if not self.data:
            print("No monitoring data collected")
            return

        # Calculate statistics
        max_cpu = max(d['process']['cpu_percent'] for d in self.data)
        avg_cpu = sum(d['process']['cpu_percent'] for d in self.data) / len(self.data)
        max_memory = max(d['process']['memory_mb'] for d in self.data)
        avg_memory = sum(d['process']['memory_mb'] for d in self.data) / len(self.data)
        max_threads = max(d['process']['num_threads'] for d in self.data)
        max_children = max(d['children']['count'] for d in self.data)

        # Find parallel execution phase (when children exist)
        parallel_phases = [d for d in self.data if d['children']['count'] > 0]
        if parallel_phases:
            parallel_start = parallel_phases[0]['timestamp']
            parallel_end = parallel_phases[-1]['timestamp']
            parallel_duration = parallel_end - parallel_start
            parallel_avg_cpu = sum(d['children']['cpu_percent'] for d in parallel_phases) / len(parallel_phases)
        else:
            parallel_duration = 0
            parallel_avg_cpu = 0

        report = {
            'summary': {
                'total_duration': self.data[-1]['timestamp'],
                'samples_collected': len(self.data),
                'max_cpu_percent': max_cpu,
                'avg_cpu_percent': avg_cpu,
                'max_memory_mb': max_memory,
                'avg_memory_mb': avg_memory,
                'max_threads': max_threads,
                'max_parallel_workers': max_children,
                'parallel_phase_duration': parallel_duration,
                'parallel_avg_cpu': parallel_avg_cpu
            },
            'timeline': self.data
        }

        # Save JSON report
        report_file = Path(output_path).with_suffix('.monitoring.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        # Print summary
        print(f"\n{'='*80}")
        print("PROCESS MONITORING SUMMARY")
        print(f"{'='*80}")
        print(f"\n📊 RESOURCE USAGE:")
        print(f"  Peak CPU:           {max_cpu:.1f}%")
        print(f"  Average CPU:        {avg_cpu:.1f}%")
        print(f"  Peak Memory:        {max_memory:.1f} MB")
        print(f"  Average Memory:     {avg_memory:.1f} MB")
        print(f"  Max Threads:        {max_threads}")

        print(f"\n⚡ PARALLELISM:")
        print(f"  Max Workers:        {max_children}")
        print(f"  Parallel Duration:  {parallel_duration:.2f}s")
        if parallel_avg_cpu > 0:
            print(f"  Parallel Avg CPU:   {parallel_avg_cpu:.1f}%")

        print(f"\n💾 MONITORING DATA:")
        print(f"  Samples Collected:  {len(self.data)}")
        print(f"  Total Duration:     {self.data[-1]['timestamp']:.2f}s")
        print(f"  Report Saved:       {report_file}")

        return report


def main():
    parser = argparse.ArgumentParser(
        description="Run parallel processing with process monitoring"
    )
    parser.add_argument("audio", type=str, help="Path to audio file")
    parser.add_argument("--hf-token", type=str, required=True, help="HuggingFace token")
    parser.add_argument("--whisper-model", type=str, default="turbo")
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--multilingual", action="store_true")
    parser.add_argument("--refine-with-llm", action="store_true", default=False)
    parser.add_argument("--vertex-project-id", type=str, default=None)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--monitoring-interval", type=float, default=0.5,
                        help="Process monitoring interval in seconds")

    args = parser.parse_args()

    # Verify audio file exists
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        sys.exit(1)

    # Determine output path
    if args.output is None:
        audio_path = Path(args.audio)
        output_path = audio_path.parent / f"{audio_path.stem}_transcription"
    else:
        output_path = args.output

    # Start process monitoring
    monitor = ProcessMonitor(interval=args.monitoring_interval)
    monitor.start()

    try:
        # Import and run parallel processing
        from parallel_whisper_pyannote import ParallelWhisperDiarization, TranscriptRefiner, save_results

        # Initialize pipeline
        print(f"\n{'='*80}")
        print("INITIALIZING PIPELINE")
        print(f"{'='*80}\n")

        pipeline = ParallelWhisperDiarization(
            whisper_model=args.whisper_model,
            diarization_model="pyannote/speaker-diarization-3.1",
            device=None,  # Auto-detect
            hf_token=args.hf_token
        )

        # Run transcription with parallel processing
        result = pipeline.transcribe(
            args.audio,
            language=args.language,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            multilingual=args.multilingual
        )

        # Save raw results
        save_results(result, output_path, "all", suffix="_raw" if args.refine_with_llm else "")

        # Apply LLM refinement if requested
        if args.refine_with_llm:
            try:
                refiner = TranscriptRefiner(
                    project_id=args.vertex_project_id,
                    region="us-east5"
                )

                refined_result = refiner.refine_transcript(result, chunk_size=20)
                save_results(refined_result, output_path, "all", suffix="_refined")
                result = refined_result

            except Exception as e:
                print(f"\nWarning: LLM refinement failed: {e}")
                print("Continuing with raw results...")

        # Print results summary
        print(f"\n{'='*80}")
        print("TRANSCRIPTION RESULTS")
        print(f"{'='*80}")
        print(f"Language: {result['language']}")
        print(f"Speakers: {', '.join(result['speakers'])}")
        print(f"Segments: {len(result['segments'])}")

        if 'timing' in result:
            timing = result['timing']
            print(f"\n⏱️  TIMING:")
            print(f"  Whisper:     {timing['whisper_time']:.2f}s")
            print(f"  Pyannote:    {timing['diarization_time']:.2f}s")
            print(f"  Parallel:    {timing['parallel_time']:.2f}s")
            print(f"  Merge:       {timing['merge_time']:.2f}s")
            print(f"  Total:       {timing['total_time']:.2f}s")
            print(f"  Time Saved:  {timing['time_saved']:.2f}s ({timing['time_saved']/(timing['whisper_time']+timing['diarization_time'])*100:.1f}%)")

    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Stop monitoring and generate report
        print(f"\n{'='*80}")
        print("STOPPING MONITORING")
        print(f"{'='*80}\n")

        monitoring_data = monitor.stop()
        report = monitor.generate_report(output_path)

    print(f"\n{'='*80}")
    print("✅ PROCESSING COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
