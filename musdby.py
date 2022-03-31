import musdb

mus = musdb.DB(root="database_wav",is_wav=False)
# print(mus[0].audio)

for track in mus:
    print(track.audio)
    print(5/0)