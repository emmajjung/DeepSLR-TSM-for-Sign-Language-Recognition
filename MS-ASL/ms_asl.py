import pandas as pd
import subprocess
import shlex
import os
import json


def process_rows(dataFrame, output_dir):
    for _, row in dataFrame.iterrows():
        vid_id = f"{row['url'][-10:]}_{str(row['start_time'])}"
        vid_path = os.path.join(output_dir)
        command = f'yt-dlp --extractor-args "youtube:player_client=web" -q -o {vid_id}.mp4 --download-sections *{row["start_time"]}-{row["end_time"]} --format 18 -P {vid_path}  {row["url"]}'
        subprocess.run(shlex.split(command))

            
if __name__ == '__main__':
    
    N_CLASSES = 25
    training_json = json.load(open('MS-ASL/MSASL_train.json'))
    training_dict = {}
    for row in training_json:
        for k, v in row.items():
            if k != 'review':
                training_dict.setdefault(k, []).append(v)
    training_df = pd.DataFrame(training_dict)
    
    val_json = json.load(open('MS-ASL/MSASL_val.json'))
    val_dict = {}
    for row in val_json:
        for k, v in row.items():
            if k != 'review':
                val_dict.setdefault(k, []).append(v)
    val_df = pd.DataFrame(val_dict)
    
    test_json = json.load(open('MS-ASL/MSASL_test.json'))
    test_dict = {}
    for row in test_json:
        for k, v in row.items():
            if k != 'review':
                test_dict.setdefault(k, []).append(v)
    test_df = pd.DataFrame(training_dict)
    
    top_n_classes = training_df.value_counts('label')[:N_CLASSES].keys().to_list()
    
    training_df = training_df[training_df['label'].isin(top_n_classes)]
    val_df = val_df[val_df['label'].isin(top_n_classes)]
    test_df = test_df[test_df['label'].isin(top_n_classes)]
    
    process_rows(training_df, 'ms_asl_train')
    process_rows(val_df, 'ms_asl_val')
    process_rows(test_df, 'ms_asl_test')
    
    # model = YOLO('preprocess/yolov8m-face.pt')
   
    # print(f'Processing time for rows {start} to {start + count - 1} is {time.time() - start_time} seconds.\n')
