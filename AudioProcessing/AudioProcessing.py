import numpy as np
import librosa.display
import librosa
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

# 그래프의 사이즈 조절
FIG_SIZE = (10, 5)


# 오디오 파형과 스펙트럼을 보여주는 함수
def view_audio(wavfile):
    # 오디오 파일의 신호와 샘플링레이트 load
    sig, sr = librosa.load(wavfile, sr=22050)
    # 오디오 파일의 신호와 총 크기 확인
    print(sig, sig.shape)

    plt.figure(figsize=FIG_SIZE)
    # 오디오의 파형을 조금 더 구분하기 좋게 보여주는 그래프
    # x축은 시간축, y축은 진폭
    librosa.display.waveplot(sig, sr, alpha=0.5)
    plt.xlabel('sec')
    plt.ylabel('amp')
    plt.title('wavefile : ' + wavfile)

    # 오디오 신호의 fft 변환
    fft = np.fft.fft(sig)
    # fft 변환은 복소수 범위이므로 magnitude를 구하기 위해 절대값 적용
    mag = np.abs(fft)
    # 진동수의 범위 설정
    freq = np.linspace(0, sr, len(mag))
    # 정확히 대칭형의 그래프가 나오므로 절반만 추출
    half_spec = mag[:int(len(mag)/2)]
    half_freq = freq[:int(len(mag)/2)]

    plt.figure(figsize=FIG_SIZE)
    # 오디오의 스펙트럼
    # x축은 진동수, y축은 magnitude
    plt.plot(half_freq, half_spec)
    plt.xlabel("freq")
    # plt.xlim(0, 1000, 100)
    # plt.xlim(4000, 5000, 100)
    plt.ylabel("mag")
    # plt.ylim(0, 15000, 1000)
    # plt.ylim(0, 1000, 100)
    plt.title("spectrum : " + wavfile)

    plt.show()


# humtest_clean.wav 파일 생성 함수
def clean_humtest():
    sig, sr = librosa.load('humtest.wav', sr=22050)
    sig_test, sr_test = librosa.load('test.wav', sr=22050)

    print(sig, sig.shape)

    fft = np.fft.fft(sig)
    fft_test = np.fft.fft(sig_test)

    # humtest.wav의 fft의 특이점 탐색
    # 대략 11000 - 12000 사이에 존재
    plt.figure(figsize=FIG_SIZE)
    plt.title('hum')
    plt.plot(fft.real * 2, 'r')

    # test.wav의 fft와 비교
    plt.title('test')
    plt.plot(fft_test.real, 'b')

    # 노이즈가 있는 부분만을 따로 추출
    noise_fft1 = fft[11000:12000]
    noise_fft2 = fft[-12000:-11000]
    noise_fft = np.zeros(shape=11000)
    noise_fft = np.append(noise_fft, noise_fft1)
    noise_fft = np.append(noise_fft, np.zeros(shape=(len(fft)-12000*2)))
    noise_fft = np.append(noise_fft, noise_fft2)
    noise_fft = np.append(noise_fft, np.zeros(shape=11000))

    # 노이즈 부분의 fft
    plt.figure(figsize=FIG_SIZE)
    plt.title('noise_fft')
    plt.plot(noise_fft.real)

    # 노이즈만 있는 부분을 빼서 노이즈를 제거
    fft = (fft - noise_fft) * 2

    # 노이즈를 제거한 fft
    plt.figure(figsize=FIG_SIZE)
    plt.title('hum_clean')
    plt.plot(fft.real)
    plt.show()

    # fft를 ifft를 취해 데이터로 변환
    audio_clean = np.fft.ifft(fft).real

    mse = np.sum((sig - audio_clean) ** 2) / len(sig)

    print('humtest MSE : ', mse)

    write('humtest_clean.wav', 22050, audio_clean.astype(np.float32))


# electest_clean.wav 파일 생성 함수 (미완)
def clean_electest():
    sig, sr = librosa.load('electest.wav', sr=22050)
    sig_test, sr_test = librosa.load('test.wav', sr=22050)

    print(sig, sig.shape)

    fft = np.fft.fft(sig)
    fft_test = np.fft.fft(sig_test)

    # electest.wav의 fft의 특이점을 탐색
    plt.figure(figsize=FIG_SIZE)
    plt.title('elec')
    # plt.xlim(0, 321760)
    # plt.ylim(-1000, 1000)
    plt.plot(fft.real, 'r')

    # test.wav의 fft와 비교
    plt.title('test')
    # plt.xlim(0, 321760)
    # plt.ylim(-1000, 1000)
    plt.plot(fft_test.real, 'b')

    noise_fft1_l = fft[1650:1800]
    noise_fft2_l = fft[5200:5300]
    noise_fft3_l = fft[97500:98000]
    noise_fft4_l = fft[101100:101500]
    noise_fft5_l = fft[104500:105500]
    noise_fft6_l = fft[111600:112000]
    noise_fft7_l = fft[118600:119000]
    noise_fft1_r = fft[-1800:-1650]
    noise_fft2_r = fft[-5300:-5200]
    noise_fft3_r = fft[-98000:-97500]
    noise_fft4_r = fft[-101500:-101100]
    noise_fft5_r = fft[-105500:-104500]
    noise_fft6_r = fft[-112000:-111600]
    noise_fft7_r = fft[-119000:-118600]
    noise_fft = np.zeros(shape=1650)
    noise_fft = np.append(noise_fft, noise_fft1_l)
    noise_fft = np.append(noise_fft, np.zeros(shape=(5200 - 1800)))
    noise_fft = np.append(noise_fft, noise_fft2_l)
    noise_fft = np.append(noise_fft, np.zeros(shape=(97500 - 5300)))
    noise_fft = np.append(noise_fft, noise_fft3_l)
    noise_fft = np.append(noise_fft, np.zeros(shape=(101100 - 98000)))
    noise_fft = np.append(noise_fft, noise_fft4_l)
    noise_fft = np.append(noise_fft, np.zeros(shape=(104500 - 101500)))
    noise_fft = np.append(noise_fft, noise_fft5_l)
    noise_fft = np.append(noise_fft, np.zeros(shape=(111600 - 105500)))
    noise_fft = np.append(noise_fft, noise_fft6_l)
    noise_fft = np.append(noise_fft, np.zeros(shape=(118600 - 112000)))
    noise_fft = np.append(noise_fft, noise_fft7_l)
    noise_fft = np.append(noise_fft, np.zeros(shape=(len(fft) - 119000 * 2)))
    noise_fft = np.append(noise_fft, noise_fft7_r)
    noise_fft = np.append(noise_fft, np.zeros(shape=(118600 - 112000)))
    noise_fft = np.append(noise_fft, noise_fft6_r)
    noise_fft = np.append(noise_fft, np.zeros(shape=(111600 - 105500)))
    noise_fft = np.append(noise_fft, noise_fft5_r)
    noise_fft = np.append(noise_fft, np.zeros(shape=(104500 - 101500)))
    noise_fft = np.append(noise_fft, noise_fft4_r)
    noise_fft = np.append(noise_fft, np.zeros(shape=(101100 - 98000)))
    noise_fft = np.append(noise_fft, noise_fft3_r)
    noise_fft = np.append(noise_fft, np.zeros(shape=(97500 - 5300)))
    noise_fft = np.append(noise_fft, noise_fft2_r)
    noise_fft = np.append(noise_fft, np.zeros(shape=(5200 - 1800)))
    noise_fft = np.append(noise_fft, noise_fft1_r)
    noise_fft = np.append(noise_fft, np.zeros(shape=1650))

    # 노이즈 부분의 fft
    plt.figure(figsize=FIG_SIZE)
    # plt.xlim(0, 321760)
    # plt.ylim(-1000, 1000)
    plt.title('noise_fft')
    plt.plot(noise_fft.real)

    fft = fft - noise_fft

    plt.figure(figsize=FIG_SIZE)
    plt.title('elec_clean')
    plt.plot(fft.real, 'r')

    plt.title('test')
    plt.xlim(100000, 110000)
    plt.ylim(-500, 500)
    plt.plot(fft_test.real, 'b')

    plt.show()

    audio_clean = np.fft.ifft(fft).real

    mse = np.sum((sig - audio_clean) ** 2) / len(sig)

    print('electest MSE : ', mse)

    write('electest_clean.wav', 22050, audio_clean.astype(np.float32))


# 오디오 파형과 스펙트럼 관찰
view_audio('test.wav')
view_audio('humtest.wav')
view_audio('electest.wav')

# 노이즈 제거 파일 생성
clean_humtest()
clean_electest()
