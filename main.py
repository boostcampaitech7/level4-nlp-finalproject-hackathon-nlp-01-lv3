video_folder = 'dataset_video_sample/'
clips_folder = ''
frames_folder = ''
output_folder = '' 


video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
print(f"총 {len(video_files)}개의 비디오 파일을 발견했습니다.")

