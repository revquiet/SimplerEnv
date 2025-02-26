'''
A simple example to demonstrate the usage of the robot controller.
'''
import argparse
import time
import numpy as np
import math
from typing import Iterator
import pickle
import rerun as rr
import rerun.blueprint as rrb

def value_generator(data: list) -> Iterator[float]:

    # for row in data:
    #      for value in row:
    #         yield value
    for value in data:
        yield value

def random_walk_generator() -> Iterator[float]:
    value = 0.0
    while True:
        value += np.random.normal()
        yield value
        
def rerun_vis(real: list, des: list ) -> None:
    parser = argparse.ArgumentParser(description="Plot dashboard stress test")
    rr.script_add_args(parser)

    parser.add_argument("--num-plots", type=int, default=7, help="How many different plots?")
    parser.add_argument("--num-series-per-plot", type=int, default=2, help="How many series in each single plot?")
    parser.add_argument("--freq", type=float, default=100, help="Frequency of logging (applies to all series)")
    parser.add_argument("--window-size", type=float, default=15.0, help="Size of the window in seconds")
    parser.add_argument("--duration", type=float, default=60, help="How long to log for in seconds")
    parser.add_argument("--config", type=str, help="Path to the configuration file")

    args = parser.parse_args()

    plot_paths = [f"plot_{i}" for i in range(0, args.num_plots)]
    series_paths = [f"series_{i}" for i in range(0, args.num_series_per_plot)]

    rr.script_setup(args, "rerun_example_live_scrolling_plot")

    # Always send the blueprint since it is a function of the data.
    rr.send_blueprint(
        rrb.Grid(
            contents=[
                rrb.TimeSeriesView(
                    origin=plot_path,
                    time_ranges=[
                        rrb.VisibleTimeRange(
                            "time",
                            start=rrb.TimeRangeBoundary.cursor_relative(seconds=-args.window_size),
                            end=rrb.TimeRangeBoundary.cursor_relative(),
                        )
                    ],
                    plot_legend=rrb.PlotLegend(visible=False),
                )
                for plot_path in plot_paths
            ]
        ),
    )

    # Generate a list of generators for each series in each plot
    values = [[random_walk_generator() for _ in range(args.num_series_per_plot)] for _ in range(args.num_plots)]
    print(len(values), len(values[0]))

    #if num_series_per_plot==2,num_plots==4
    # [
    #     [0, 1],
    #     [2, 5],
    #     [2, 5],
    #     [2, 5],
    # ]
    real_columns = [[row[i] for row in real] for i in range(len(real[0]))]
    des_columns = [[row[i] for row in des] for i in range(len(des[0]))]

    values = [
        [value_generator(real_columns[count]) ,value_generator(des_columns[count])]
        for count in range(args.num_plots)
    ]
    print(len(values), len(values[0]))
    cur_time = time.time()
    end_time = cur_time + args.duration
    time_per_tick = 1.0 / args.freq
    try:
        while cur_time < end_time:
            # Advance time and sleep if necessary
            cur_time += time_per_tick
            sleep_for = cur_time - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)

            if sleep_for < -0.1:
                print(f"Warning: missed logging window by {-sleep_for:.2f} seconds")

            rr.set_time_seconds("time", cur_time)

            # Output each series based on its generator
            for plot_idx, plot_path in enumerate(plot_paths):
                for series_idx, series_path in enumerate(series_paths):
                    rr.log(f"{plot_path}/{series_path}", rr.Scalar(next(values[plot_idx][series_idx])))
    except StopIteration:
        pass

    rr.script_teardown(args)

def write_positions_to_file(position_desired_list, position_real_list, file_path):
    """
    使用 pickle 将 position_desired_list 和 position_real_list 写入文件。
    """
    data = {
        "position_desired_list": position_desired_list,
        "position_real_list": position_real_list,
    }
    with open(file_path, "ab") as file:
        pickle.dump(data, file)
    print(f"Data successfully written to {file_path}")

def load_file(file_path):
    """
    从文件中读取 position_desired_list 和 position_real_list，并返回它们。
    """
    position_desired_list = []
    position_real_list = []
    
    with open(file_path, "rb") as file:  # 使用 "rb" 模式读取二进制文件
        try:
            while True:  # 循环读取文件中的所有数据
                data = pickle.load(file)
                position_desired_list.extend(data["position_desired_list"])
                position_real_list.extend(data["position_real_list"])
        except EOFError:
            # 当文件结束时会抛出 EOFError，正常退出循环
            pass
    
    return position_desired_list, position_real_list

if __name__ == "__main__":
    # main()

    # #TODO current
    position_desired_list, position_real_list = load_file("/home/yangqj/Code/git_repo/fourier-grx-dds/examples/impedance_current1.pkl")
    # position_desired_list, position_real_list = load_file("/home/yangqj/Code/git_repo/fourier-grx-dds/examples/impedance_position1.pkl")
    rerun_vis(position_desired_list, position_real_list)