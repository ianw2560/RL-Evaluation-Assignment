import pandas as pd
import glob

metrics_csv_files = glob.glob("metrics/*metrics.csv")

for csv in metrics_csv_files:

    df = pd.read_csv(csv)

    pd.set_option('display.precision', 4)
    pd.options.display.float_format = '{:,.4f}'.format

    # Build metrics table title
    title = csv.split("/")[-1]
    title = title.split(".")[0]

    # Get testname in order to determine what additional column to print
    test_name = title.split("_")
    test_name = test_name[0]

    title = title.replace("_", " ").title()

    # Map parameter name
    parameter = {
        "entropy": "EntCoef",
        "batchsize": "BatchSize",
        "learningrate": "LearningRate",
        "episodelength": "EpisodeLength",
    }

    print(f"Table: {title}")

    if test_name == "best":
        print(df[["Algorithm", "MAE", "AvgJerk", "VarianceJerk"]].to_string(index=False))
    else:
        print(df[["Algorithm", parameter[test_name], "MAE", "AvgJerk", "VarianceJerk"]].to_string(index=False))
    print()
    