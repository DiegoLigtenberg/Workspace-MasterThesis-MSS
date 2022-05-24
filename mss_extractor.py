import mss

# generate post processed dataset
separator = mss.Separator(post_processing=True)
separator.input_to_waveform()

# # generate raw model output dataset
# separator.post_processing = False
# separator.input_to_waveform()



'''
For each track in track_input

1) Load track
2) preprocess the track 
3) Model the track

'''


