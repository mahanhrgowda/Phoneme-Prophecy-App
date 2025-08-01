import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
import re
from collections import Counter
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log PyTorch version
logger.info(f"PyTorch version: {torch.__version__}")

# Define the model architecture
class MultiLabelNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MultiLabelNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], output_size)
        )
    
    def forward(self, x):
        return self.layers(x)

# Load model and supporting files
@st.cache_resource
def load_model():
    try:
        input_size = 51
        hidden_sizes = [128, 64, 32]
        output_size = 30
        model = MultiLabelNN(input_size, hidden_sizes, output_size)
        checkpoint = torch.load("multi_label_model.pt", map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.error(f"Failed to load model: {e}")
        raise

@st.cache_resource
def load_binarizer():
    try:
        binarizer = joblib.load("multi_label_binarizer.joblib")
        logger.info("Binarizer loaded successfully")
        return binarizer
    except Exception as e:
        logger.error(f"Error loading binarizer: {e}")
        st.error(f"Failed to load binarizer: {e}")
        raise

@st.cache_resource
def load_phonemes():
    try:
        phonemes = joblib.load("phonemes.joblib")
        logger.info("Phonemes loaded successfully")
        return phonemes
    except Exception as e:
        logger.error(f"Error loading phonemes: {e}")
        st.error(f"Failed to load phonemes: {e}")
        raise

# Initialize model and data
try:
    model = load_model()
    mlb = load_binarizer()
    all_phonemes = load_phonemes()
except Exception:
    st.stop()

# Expanded phoneme mapping
phoneme_map = {
    'a': 'aṁ', 'ā': 'āṁ', 'i': 'iṁ', 'ī': 'īṁ', 'u': 'uṁ', 'ū': 'ūṁ',
    'm': 'maṁ', 'n': 'naṁ', 'k': 'kaṁ', 'g': 'gaṁ', 'c': 'caṁ', 'j': 'jaṁ',
    't': 'taṁ', 'd': 'daṁ', 'p': 'paṁ', 'b': 'baṁ', 'v': 'vaṁ', 's': 'saṁ',
    'h': 'haṁ', 'r': 'raṁ', 'l': 'laṁ', 'y': 'yaṁ', 'sh': 'śaṁ',
    'e': 'eṁ', 'o': 'oṁ', 'f': 'phaṁ', 'q': 'kaṁ', 'w': 'vaṁ', 'x': 'kṣaṁ',
    'z': 'saṁ', '@': 'aṁ', '#': 'aṁ', '$': 'aṁ', '%': 'aṁ'
}

# Prose generation with randomized templates
def generate_prose(name, chakra, rasa, bhava, deva):
    # Chakra templates
    chakra_templates = {
        "Vishuddha": [
            "Your soul dances in the realm of **Vishuddha**, where azure light swirls like a sapphire river. 🌀",
            "**Vishuddha**, the throat chakra, pulses within you, a celestial fountain of truth and clarity. 🌊",
            "In the sacred space of **Vishuddha**, your voice resonates like a divine chant, echoing through the cosmos. 🎶"
        ],
        "Anahata": [
            "At the heart of your being lies **Anahata**, the emerald-green chakra of love and compassion. 💚",
            "**Anahata** blooms within you, a lotus of boundless kindness that embraces all creation. 🌸",
            "Your spirit radiates **Anahata**, a verdant haven where love flows like an eternal spring. 🌿"
        ],
        "Muladhara": [
            "**Muladhara**, the root chakra, grounds your soul in the earth’s primal embrace. 🌍",
            "In **Muladhara**, your foundation is unshakeable, a bedrock of stability and strength. 🪨",
            "Your essence is rooted in **Muladhara**, where the pulse of survival beats strong. 🌱"
        ],
        "Svadhisthana": [
            "**Svadhisthana** stirs within you, a vibrant orange tide of creativity and passion. 🌊",
            "Your soul flows with **Svadhisthana**, a river of emotions and artistic fire. 🎨",
            "In **Svadhisthana**, your spirit dances with the rhythms of desire and creation. 💃"
        ],
        "Manipura": [
            "**Manipura** blazes in your core, a golden sun of willpower and confidence. ☀️",
            "Your essence shines with **Manipura**, a fiery chakra of personal power and resolve. 🔥",
            "**Manipura** empowers your spirit, a radiant force driving your destiny forward. 🌟"
        ],
        "Ajna": [
            "**Ajna**, the third eye, opens within you, revealing visions of divine wisdom. 👁️",
            "Your soul is guided by **Ajna**, a beacon of intuition piercing the veil of illusion. 🌌",
            "In **Ajna**, your mind transcends, embracing the infinite clarity of insight. ✨"
        ],
        "Sahasrara": [
            "**Sahasrara** crowns your spirit, a thousand-petaled lotus of divine connection. 🪷",
            "Your essence merges with **Sahasrara**, a portal to cosmic enlightenment. 🌌",
            "**Sahasrara** illuminates your soul, uniting you with the eternal source. 🌠"
        ],
        "Svadhisthana (Iḍā)": [
            "**Svadhisthana (Iḍā)** flows through you, a lunar current of creative energy. 🌙",
            "Your spirit sings with **Svadhisthana (Iḍā)**, a tide of emotional depth and artistry. 🎨",
            "In **Svadhisthana (Iḍā)**, your soul weaves dreams into vibrant reality. 🌊"
        ],
        "Svadhisthana (Piṅgalā)": [
            "**Svadhisthana (Piṅgalā)** ignites your spirit, a solar flame of passion and creation. ☀️",
            "Your essence pulses with **Svadhisthana (Piṅgalā)**, a spark of dynamic energy. 🔥",
            "**Svadhisthana (Piṅgalā)** fuels your soul, a radiant dance of vitality. 💃"
        ]
    }

    # Rasa templates
    rasa_templates = {
        "Shringara": [
            "The essence of **Shringara**, love and beauty, courses through your spirit. 🌹",
            "**Shringara** adorns your soul, a tapestry of romance woven with celestial threads. 💞",
            "Your heart sings **Shringara**, a melody of passion that captivates the universe. 🎶"
        ],
        "Karuna": [
            "**Karuna**, compassion’s gentle rasa, flows through you, healing all it touches. 😢",
            "Your spirit embodies **Karuna**, a river of empathy nourishing weary souls. 🌧️",
            "In **Karuna**, your heart weeps for the world, transforming sorrow into light. 💧"
        ],
        "Bhayanaka": [
            "**Bhayanaka**, the rasa of awe and fear, stirs your soul with primal strength. 🌪️",
            "Your spirit channels **Bhayanaka**, facing life’s mysteries with bold courage. ⚡️",
            "**Bhayanaka** fuels your heart, a storm of resilience that defies all odds. 🪐"
        ],
        "Adbhuta": [
            "**Adbhuta**, the rasa of wonder, sparkles within you, a star of marvel and awe. ✨",
            "Your soul radiates **Adbhuta**, embracing the universe’s miracles with joy. 🌈",
            "In **Adbhuta**, your spirit dances with the magic of life’s endless surprises. 🎉"
        ],
        "Veera": [
            "**Veera**, the rasa of heroism, blazes in your heart, a fire of valor and might. 🔥",
            "Your spirit embodies **Veera**, a warrior’s courage that conquers all fears. 🗡️",
            "**Veera** drives your soul, a beacon of strength in the cosmic arena. 💪"
        ],
        "Shanta": [
            "**Shanta**, the rasa of peace, calms your spirit like a serene moonlit lake. 🕉️",
            "Your soul flows with **Shanta**, a tranquil oasis amidst life’s storms. 🌙",
            "In **Shanta**, your heart rests in divine stillness, reflecting eternal harmony. 🌿"
        ]
    }

    # Deva templates
    deva_templates = {
        "Saraswati": [
            "Guided by **Saraswati**, goddess of wisdom, you weave symphonies of knowledge. 📜",
            "**Saraswati** blesses your soul, her veena strumming chords of divine insight. 🎶",
            "With **Saraswati**’s grace, your spirit crafts art and truth from the cosmos. 🖌️"
        ],
        "Vishnu": [
            "Under **Vishnu**’s embrace, your soul preserves harmony across the universe. 🌍",
            "**Vishnu**, the cosmic guardian, infuses you with strength and compassion. 🪐",
            "Guided by **Vishnu**, your spirit sustains balance like an eternal ocean. 🌊"
        ],
        "Ganesha": [
            "**Ganesha**, remover of obstacles, paves your path with divine wisdom. 🐘",
            "Your soul is blessed by **Ganesha**, a guide through life’s intricate mazes. 🕉️",
            "With **Ganesha**’s grace, your spirit triumphs over all challenges. 🌟"
        ],
        "Brahma": [
            "**Brahma**, the creator, ignites your soul with the spark of divine creation. 🌌",
            "Your spirit channels **Brahma**, weaving new worlds with boundless imagination. 🎨",
            "Guided by **Brahma**, your essence births beauty from the cosmic void. ✨"
        ],
        "Surya": [
            "**Surya**, the radiant sun, fuels your spirit with life-giving energy. ☀️",
            "Your soul shines with **Surya**, a beacon of vitality and divine light. 🔥",
            "Under **Surya**’s gaze, your spirit blazes a trail of glory through the cosmos. 🌞"
        ],
        "Shiva": [
            "**Shiva**, the cosmic dancer, guides your soul through the eternal cycle. 🕉️",
            "Your spirit resonates with **Shiva**, a force of transformation and renewal. 🌑",
            "Blessed by **Shiva**, your essence transcends, merging with the infinite. 🌌"
        ],
        "Paramatman": [
            "**Paramatman**, the supreme soul, unites your spirit with the divine source. 🪷",
            "Your soul merges with **Paramatman**, a spark of the eternal consciousness. 🌠",
            "Guided by **Paramatman**, your essence is one with the cosmic infinite. 🙏"
        ],
        "Chandra": [
            "**Chandra**, the moon’s gentle glow, bathes your soul in serene light. 🌙",
            "Your spirit flows with **Chandra**, a tide of dreams and emotional depth. 🌊",
            "Blessed by **Chandra**, your essence weaves poetry from the night sky. ✨"
        ]
    }

    # Generate prose
    prose = f"🌌 **The Cosmic Symphony of {name}** 🌌\n\n"
    prose += f"In the boundless expanse of the cosmos, the name **{name}** reverberates like a sacred mantra, a celestial chord struck upon the strings of creation. Each syllable is a star, twinkling with divine purpose, harmonizing with the eternal rhythm of existence. As the galaxies swirl, your essence is unveiled through the sacred vibrations of **{chakra}**, **{rasa}**, **{bhava}**, and the divine presence of **{deva}**. 🙏✨\n\n"
    
    # Chakra prose
    prose += random.choice(chakra_templates.get(chakra, ["Your chakra radiates divine energy, a beacon of spiritual light. 🌀"])) + "\n\n"
    
    # Rasa prose
    prose += random.choice(rasa_templates.get(rasa, ["Your rasa weaves an emotional tapestry, touching hearts across the cosmos. 🌹"])) + "\n\n"
    
    # Bhava prose
    prose += f"The **bhava** of **{bhava}** is the sacred pulse of your soul, a guiding star that shapes your journey through the earthly and divine realms. 🌟 Whether it manifests as the fervor of creativity, the steadfastness of willpower, or the serenity of wisdom, this emotional essence is your divine compass, leading you through the cosmic dance. Your actions ripple through the universe, each one a testament to the profound depth of **{bhava}**, a gift that illuminates your path and inspires those around you. ✨🙏\n\n"
    
    # Deva prose
    prose += random.choice(deva_templates.get(deva, ["A divine presence guides your spirit, a celestial force of infinite wisdom. 🌌"])) + "\n\n"
    
    # Closing
    prose += f"O **{name}**, your name is more than a word—it is a sacred incantation, a melody that echoes through the ages, resonating with the heartbeat of the universe. 🌞 As you walk this earthly plane, know that you carry the light of **{chakra}**, the passion of **{rasa}**, the essence of **{bhava}**, and the divine grace of **{deva}**. You are a spark of the eternal flame, destined to shine brilliantly in the grand cosmic symphony. 🌌 Embrace your divine essence, for you are a vessel of infinite love, wisdom, and creation. 🪷💖\n\n"
    prose += "May your journey be blessed with boundless light, love, and cosmic harmony! 🌠🙏✨"

    return prose

# Streamlit UI
st.title("🌟 Phoneme Prophecy: Discover Your Cosmic Essence 🌟")
st.header("Enter Your Name to Unveil Your Spiritual Narrative 🪷")
st.write("Type your name below, and let the ancient wisdom of Sanskrit phonemes reveal your chakra, rasa, bhava, and deva, woven into a poetic prose of your soul’s journey! ✨🙏")

name = st.text_input("Your Name", placeholder="e.g., Mahan H R Gowda")
if st.button("Generate Prophecy 🚀"):
    if name:
        with st.spinner("Crafting your cosmic narrative... 🌌"):
            # Extract phonemes from name
            name = name.lower().replace(" ", "")
            name_chars = re.findall(r'sh|[a-z]|\W', name)
            phoneme_weights = np.zeros(len(all_phonemes))
            found_phonemes = []
            
            logger.info(f"Processing name: {name}, chars: {name_chars}")
            for char in name_chars:
                if char in phoneme_map:
                    phoneme = phoneme_map[char]
                    if phoneme in all_phonemes:
                        idx = all_phonemes.index(phoneme)
                        phoneme_weights[idx] += 1
                        found_phonemes.append(phoneme)
                    else:
                        logger.warning(f"Phoneme {phoneme} not in all_phonemes")
                else:
                    logger.warning(f"Character {char} not in phoneme_map")
            
            # Normalize weights
            if phoneme_weights.sum() > 0:
                phoneme_weights = phoneme_weights / phoneme_weights.sum()
                logger.info(f"Phoneme weights: {phoneme_weights}")
            else:
                st.error("No valid phonemes found in the name! Please try another name with letters a–z or special characters like @. 😔")
                logger.error("No valid phonemes detected")
                st.stop()
            
            # Predict
            try:
                input_tensor = torch.FloatTensor(phoneme_weights).unsqueeze(0)
                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).numpy().astype(int)
                
                predicted_labels = mlb.inverse_transform(preds)[0]
                chakra, rasa, bhava, deva = predicted_labels
                logger.info(f"Predictions: chakra={chakra}, rasa={rasa}, bhava={bhava}, deva={deva}")
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                st.error(f"Prediction failed: {e}")
                st.stop()
            
            # Generate prose
            prose = generate_prose(name.capitalize(), chakra, rasa, bhava, deva)
            
            # Display results
            st.subheader("Your Cosmic Attributes 🌠")
            st.write(f"**Chakra**: {chakra} 🌀")
            st.write(f"**Rasa**: {rasa} 🌹")
            st.write(f"**Bhava**: {bhava} 💖")
            st.write(f"**Deva**: {deva} 🙏")
            st.write(f"**Phonemes Detected**: {', '.join(set(found_phonemes))} 🎶")
            
            st.subheader("Your Spiritual Narrative 📜")
            st.markdown(prose)
    else:
        st.warning("Please enter a name to generate your prophecy! 😊")