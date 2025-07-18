# Detection System Voice Spoofing Attacks

## Setup & How to Use
**1.** PIP Install the component (download from PyPI)
```
pip install -r requirements.txt

streamlit run src/demo.py


```
**2.** Import and Initialize the component (at the top of your script)
```python
from st_audiorec import st_audiorec
```
**3.** Add an Instance of the audio recorder to your streamlit app's code.
```python 
wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    st.audio(wav_audio_data, format='audio/wav')
```
**4. Enjoy recording audio inside your streamlit app! 🎈**

Feel free to reach out to me in case you have any questions! <br>
Pls consider leaving a `star` ☆ with this repository to show your support.
