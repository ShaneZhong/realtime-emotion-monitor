import matplotlib.pyplot as plt
from pygame import mixer
from datetime import timedelta

def plot_emotion_charts(data, start_datetime, axs):
    # Fetch emotion list
    exp_list = [i[2] for i in data]
    list_angry = [i[0] for i in exp_list]
    list_happy = [i[1] for i in exp_list]
    list_neutral = [i[2] for i in exp_list]
    list_sad = [i[3] for i in exp_list]
    list_surprise = [i[4] for i in exp_list]
    list_time = [start_datetime + timedelta(seconds=x) for x in range(len(list_happy))]

    # Generate charts
    axs[0].clear()
    axs[1].clear()
    axs[2].clear()
    axs[3].clear()
    axs[4].clear()
    axs[0].plot(list_time, list_neutral, color='gray', linewidth=2, label='Neutral')
    axs[1].plot(list_time, list_happy, color='skyblue', linewidth=2, label='Happy')
    axs[2].plot(list_time, list_surprise, color='green', linewidth=2, label='Surprise')
    axs[3].plot(list_time, list_angry, color='red', linewidth=2, label='Angry')
    axs[4].plot(list_time, list_sad, color='orange', linewidth=2, label='Sad')
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[3].legend()
    axs[4].legend()
    plt.draw()
    plt.pause(0.5)


def trigger_sound(sound_dir, sound_time_interval, data):
    exp_list = [i[2] for i in data]
    list_angry = [i[0] for i in exp_list]
    list_sad = [i[3] for i in exp_list]
    avg_sad = sum(list_sad[-sound_time_interval:]) / sound_time_interval
    avg_angry = sum(list_angry[-sound_time_interval:]) / sound_time_interval
    if avg_sad >= 0.5 or avg_angry >= 0.5:
        print("Trigger Sound - Relax :)")
        mixer.init()
        mixer.music.load(sound_dir)
        mixer.music.play()