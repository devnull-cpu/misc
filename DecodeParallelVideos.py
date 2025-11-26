#!/usr/bin/env python3
"""
Decode multiple different videos in parallel using threading.

This script demonstrates the most efficient way to decode many videos at once:
- Uses threading for minimal overhead (shared CUDA context)
- Pre-allocates decoder sessions with SetSessionCount()
- Each thread decodes a different video file
- Supports batch frame processing for maximum throughput
"""

import sys
import os
import argparse
from pathlib import Path
import pycuda.driver as cuda
import PyNvVideoCodec as nvc
import numpy as np
import threading
import time
from os.path import join, dirname, abspath
from queue import Queue

# Force unbuffered output for real-time progress display
if sys.stdout is not None:
    sys.stdout.reconfigure(line_buffering=True)

# Add the parent directory to Python path
current_dir = dirname(abspath(__file__))
parent_dir = dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Thread-safe performance tracking
stats_lock = threading.Lock()
total_videos_decoded = 0
total_frames_decoded = 0
total_decode_time = 0.0
threads_started = 0
threads_completed = 0


def decode_video_thread(thread_id, cuda_ctx, cuda_stream, gpu_id, video_path,
                       save_output=False, output_dir=None, batch_size=1,
                       use_threaded_decoder=False, buffer_size=12):
    """
    Decode a single video in a thread.

    Parameters:
        thread_id (int): Thread identifier for logging
        cuda_ctx: Shared CUDA context
        cuda_stream: CUDA stream for this thread
        gpu_id (int): GPU device ID
        video_path (str): Path to video file to decode
        save_output (bool): Whether to save decoded frames
        output_dir (str): Directory to save output frames
        batch_size (int): Number of frames to process in batch (1 = single frame mode)
        use_threaded_decoder (bool): Use ThreadedDecoder instead of regular decoder
        buffer_size (int): Buffer size for ThreadedDecoder (default: 12)

    Returns:
        dict: Statistics about the decode operation
    """
    global total_videos_decoded, total_frames_decoded, total_decode_time, threads_started, threads_completed

    # Mark thread as started
    with stats_lock:
        threads_started += 1

    decoder_type = "ThreadedDecoder" if use_threaded_decoder else "Decoder"
    print(f"[Thread {thread_id:2d}] Starting ({decoder_type}): {os.path.basename(video_path)}", flush=True)
    sys.stdout.flush()

    cuda_ctx.push()

    stats = {
        'video': os.path.basename(video_path),
        'frames': 0,
        'time': 0.0,
        'fps': 0.0,
        'success': False,
        'error': None
    }

    try:
        # Check if file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if use_threaded_decoder:
            # Use ThreadedDecoder - has internal thread and buffering
            nv_dec = nvc.ThreadedDecoder(
                video_path,
                buffer_size=buffer_size,
                gpu_id=gpu_id,
                cuda_context=cuda_ctx.handle,
                cuda_stream=cuda_stream.handle,
                use_device_memory=True
            )
            nv_dmx = None  # ThreadedDecoder doesn't need separate demuxer
        else:
            # Use regular decoder
            nv_dmx = nvc.CreateDemuxer(filename=video_path)
            nv_dec = nvc.CreateDecoder(
                gpuid=gpu_id,
                codec=nv_dmx.GetNvCodecId(),
                cudacontext=cuda_ctx.handle,
                cudastream=cuda_stream.handle,
                usedevicememory=True
            )

        num_decoded_frames = 0
        start_time = time.perf_counter()
        last_print_time = start_time
        last_frame_count = 0  # Track frames at last print for instantaneous FPS
        print_interval = 0.5  # Print every 0.5 seconds

        # Setup output file if saving
        output_file = None
        if save_output and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_decoded.yuv")
            output_file = open(output_path, "wb")

        # Decode loop - different for ThreadedDecoder vs regular decoder
        if use_threaded_decoder:
            # ThreadedDecoder uses get_batch_frames() API
            while True:
                frames = nv_dec.get_batch_frames(batch_size if batch_size > 1 else 1)

                if len(frames) == 0:
                    break

                num_decoded_frames += len(frames)

                # Save frames if requested
                if output_file:
                    for frame in frames:
                        frame_size = frame.framesize()
                        raw_frame = np.ndarray(shape=frame_size, dtype=np.uint8)
                        cuda.memcpy_dtoh(raw_frame, frame.GetPtrToPlane(0))
                        output_file.write(bytearray(raw_frame))

                # Print progress periodically
                current_time = time.perf_counter()
                if current_time - last_print_time >= print_interval:
                    elapsed = current_time - start_time
                    time_delta = current_time - last_print_time

                    # Average FPS (from start)
                    avg_fps = num_decoded_frames / elapsed if elapsed > 0 else 0

                    # Instantaneous FPS (since last print)
                    frames_delta = num_decoded_frames - last_frame_count
                    inst_fps = frames_delta / time_delta if time_delta > 0 else 0

                    print(f"[Thread {thread_id:2d}] {os.path.basename(video_path):30s} | {num_decoded_frames:5d} frames | avg: {avg_fps:5.1f} fps | cur: {inst_fps:5.1f} fps", flush=True)
                    sys.stdout.flush()

                    last_print_time = current_time
                    last_frame_count = num_decoded_frames
        else:
            # Regular decoder uses demuxer packets
            for packet in nv_dmx:
                frames = nv_dec.Decode(packet)

                if batch_size > 1:
                    # Batch processing - collect frames
                    frame_batch = list(frames)
                    num_decoded_frames += len(frame_batch)

                    # Process batch (example: save frames)
                    if output_file:
                        for frame in frame_batch:
                            frame_size = frame.framesize()
                            raw_frame = np.ndarray(shape=frame_size, dtype=np.uint8)
                            cuda.memcpy_dtoh(raw_frame, frame.GetPtrToPlane(0))
                            output_file.write(bytearray(raw_frame))
                else:
                    # Single frame processing
                    for frame in frames:
                        num_decoded_frames += 1

                        if output_file:
                            frame_size = frame.framesize()
                            raw_frame = np.ndarray(shape=frame_size, dtype=np.uint8)
                            cuda.memcpy_dtoh(raw_frame, frame.GetPtrToPlane(0))
                            output_file.write(bytearray(raw_frame))

                # Print progress periodically INSIDE the loop
                current_time = time.perf_counter()
                if current_time - last_print_time >= print_interval:
                    elapsed = current_time - start_time
                    time_delta = current_time - last_print_time

                    # Average FPS (from start)
                    avg_fps = num_decoded_frames / elapsed if elapsed > 0 else 0

                    # Instantaneous FPS (since last print)
                    frames_delta = num_decoded_frames - last_frame_count
                    inst_fps = frames_delta / time_delta if time_delta > 0 else 0

                    print(f"[Thread {thread_id:2d}] {os.path.basename(video_path):30s} | {num_decoded_frames:5d} frames | avg: {avg_fps:5.1f} fps | cur: {inst_fps:5.1f} fps", flush=True)
                    sys.stdout.flush()

                    last_print_time = current_time
                    last_frame_count = num_decoded_frames

        # Synchronize to ensure all decoding is complete
        cuda_stream.synchronize()

        elapsed_time = time.perf_counter() - start_time
        fps = num_decoded_frames / elapsed_time if elapsed_time > 0 else 0

        # Update stats
        stats['frames'] = num_decoded_frames
        stats['time'] = elapsed_time
        stats['fps'] = fps
        stats['success'] = True

        # Update global stats (thread-safe)
        with stats_lock:
            total_videos_decoded += 1
            total_frames_decoded += num_decoded_frames
            total_decode_time += elapsed_time

        msg = f"[Thread {thread_id:2d}] ✓ {os.path.basename(video_path):40s} | {num_decoded_frames:5d} frames | {elapsed_time:6.2f}s | {fps:6.1f} fps"
        print(msg, flush=True)
        sys.stdout.flush()  # Extra flush for Windows

        if output_file:
            output_file.close()

    except nvc.PyNvVCExceptionUnsupported as e:
        stats['error'] = f"Codec not supported: {e}"
        print(f"[Thread {thread_id:2d}] ✗ {os.path.basename(video_path):40s} | ERROR: {stats['error']}", flush=True)
    except FileNotFoundError as e:
        stats['error'] = str(e)
        print(f"[Thread {thread_id:2d}] ✗ {os.path.basename(video_path):40s} | ERROR: File not found", flush=True)
    except Exception as e:
        stats['error'] = str(e)
        print(f"[Thread {thread_id:2d}] ✗ {os.path.basename(video_path):40s} | ERROR: {e}", flush=True)
    finally:
        cuda_ctx.pop()

    return stats


def decode_videos_parallel(video_paths, num_threads, gpu_id, save_output=False,
                           output_dir=None, batch_size=1, use_threaded_decoder=False,
                           buffer_size=12):
    """
    Decode multiple videos in parallel using threading.

    Parameters:
        video_paths (list): List of video file paths to decode
        num_threads (int): Number of concurrent decoding threads
        gpu_id (int): GPU device ID
        save_output (bool): Whether to save decoded frames
        output_dir (str): Directory to save output frames
        batch_size (int): Frames per batch for processing
        use_threaded_decoder (bool): Use ThreadedDecoder instead of regular decoder
        buffer_size (int): Buffer size for ThreadedDecoder
    """
    global total_videos_decoded, total_frames_decoded, total_decode_time, threads_started, threads_completed

    # Reset global stats
    total_videos_decoded = 0
    total_frames_decoded = 0
    total_decode_time = 0.0
    threads_started = 0
    threads_completed = 0

    num_videos = len(video_paths)

    decoder_type = "ThreadedDecoder" if use_threaded_decoder else "Decoder"
    print(f"\n{'='*80}")
    print(f"Parallel Video Decoder")
    print(f"{'='*80}")
    print(f"Videos to decode: {num_videos}")
    print(f"Concurrent threads: {num_threads}")
    print(f"GPU ID: {gpu_id}")
    print(f"Decoder type: {decoder_type}")
    if use_threaded_decoder:
        print(f"Buffer size: {buffer_size}")
    print(f"Batch size: {batch_size}")
    print(f"Save output: {save_output}")
    print(f"{'='*80}")

    # Show first few video paths for verification
    if num_videos > 0:
        print(f"\nFirst few videos:")
        for i, vp in enumerate(video_paths[:min(3, num_videos)]):
            print(f"  {i+1}. {vp}")
        if num_videos > 3:
            print(f"  ... and {num_videos - 3} more")
    print()

    try:
        # Initialize CUDA
        cuda.init()
        cuda_device = cuda.Device(gpu_id)
        cuda_ctx = cuda_device.make_context()

        # Create separate CUDA stream for each thread to avoid serialization
        cuda_streams = [cuda.Stream() for _ in range(num_threads)]
        print(f"Created {num_threads} CUDA streams (one per thread)\n", flush=True)

        # Pre-allocate decoder sessions
        nvc.PyNvDecoder.SetSessionCount(num_threads)
        print(f"Pre-allocated {num_threads} decoder sessions\n", flush=True)

        threads = []
        all_stats = []
        overall_start = time.perf_counter()

        # Wrapper function to collect stats
        def thread_wrapper(tid, vpath, stream):
            result = decode_video_thread(tid, cuda_ctx, stream, gpu_id, vpath,
                                        save_output, output_dir, batch_size,
                                        use_threaded_decoder, buffer_size)
            with stats_lock:
                all_stats.append(result)

        # Launch threads - each gets its own CUDA stream
        for i, video_path in enumerate(video_paths[:num_threads]):
            t = threading.Thread(
                target=thread_wrapper,
                args=(i, video_path, cuda_streams[i])  # Pass dedicated stream
            )
            t.start()
            threads.append(t)

        # Wait for all threads to complete
        for t in threads:
            t.join()

        overall_elapsed = time.perf_counter() - overall_start

        cuda_ctx.pop()
        cuda_ctx.detach()

        # Print summary
        print(f"\n{'='*80}")
        print(f"Summary")
        print(f"{'='*80}")
        print(f"Total videos processed: {total_videos_decoded}/{num_videos}")
        print(f"Total frames decoded: {total_frames_decoded:,}")
        print(f"Total decode time: {total_decode_time:.2f}s")
        print(f"Overall elapsed time: {overall_elapsed:.2f}s")

        if total_decode_time > 0:
            print(f"Average FPS per thread: {total_frames_decoded/total_decode_time:.1f}")
        if overall_elapsed > 0:
            print(f"Throughput: {num_videos/overall_elapsed:.2f} videos/sec")

        # Show failures if any
        failures = [s for s in all_stats if not s['success']]
        if failures:
            print(f"\nFailed videos: {len(failures)}")
            for s in failures:
                print(f"  - {s['video']}: {s['error']}")

        print(f"{'='*80}\n")

    except Exception as e:
        print(f"Error in parallel decode: {e}")
        if 'cuda_ctx' in locals():
            cuda_ctx.pop()
            cuda_ctx.detach()


def read_video_list(file_path):
    """Read video paths from a text file (one path per line)."""
    video_paths = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                video_paths.append(line)
    return video_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decode multiple different videos in parallel using NVIDIA GPU acceleration.\n\n"
                   "This is optimized for maximum throughput when processing many videos.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="Text file containing video paths (one per line) or single video path"
    )

    parser.add_argument(
        "-n", "--num_threads",
        type=int,
        default=64,
        help="Number of concurrent decoding threads (default: 64)\n"
             "Note: GPU session limit is typically 32-64 depending on hardware"
    )

    parser.add_argument(
        "-g", "--gpu_id",
        type=int,
        default=0,
        help="GPU device ID (default: 0)"
    )

    parser.add_argument(
        "-s", "--save_output",
        action="store_true",
        help="Save decoded frames to YUV files"
    )

    parser.add_argument(
        "-o", "--output_dir",
        type=Path,
        default="decoded_output",
        help="Directory to save decoded frames (default: decoded_output)"
    )

    parser.add_argument(
        "-b", "--batch_size",
        type=int,
        default=1,
        help="Number of frames to process in batch (default: 1)"
    )

    parser.add_argument(
        "-t", "--threaded_decoder",
        action="store_true",
        help="Use ThreadedDecoder instead of regular decoder for better pipelining"
    )

    parser.add_argument(
        "--buffer_size",
        type=int,
        default=12,
        help="Buffer size for ThreadedDecoder (default: 12)"
    )

    args = parser.parse_args()

    # Read video list
    input_path = args.input.as_posix()
    if input_path.endswith('.txt'):
        video_paths = read_video_list(input_path)
        if not video_paths:
            print(f"Error: No video paths found in {input_path}")
            sys.exit(1)
    else:
        # Single video file
        video_paths = [input_path]

    # Decode videos
    decode_videos_parallel(
        video_paths,
        args.num_threads,
        args.gpu_id,
        args.save_output,
        args.output_dir.as_posix() if args.save_output else None,
        args.batch_size,
        args.threaded_decoder,
        args.buffer_size
    )
