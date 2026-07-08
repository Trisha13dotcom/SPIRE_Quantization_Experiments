from speech_quantization import quantize
from datasets import load_dataset
import numpy as np
import librosa
from speechalgo.features import DeltaFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# All 102 FLEURS language configs with display names
FLEURS_LANGUAGES = [
    ("af_za", "Afrikaans"),     ("am_et", "Amharic"),       ("ar_eg", "Arabic"),
    ("as_in", "Assamese"),      ("ast_es", "Asturian"),      ("az_az", "Azerbaijani"),
    ("be_by", "Belarusian"),    ("bg_bg", "Bulgarian"),      ("bn_in", "Bengali"),
    ("bs_ba", "Bosnian"),       ("ca_es", "Catalan"),        ("ceb_ph", "Cebuano"),
    ("ckb_iq", "Kurdish"),      ("cmn_hans_cn", "Chinese"),  ("cs_cz", "Czech"),
    ("cy_gb", "Welsh"),         ("da_dk", "Danish"),         ("de_de", "German"),
    ("el_gr", "Greek"),         ("en_us", "English"),        ("es_419", "Spanish"),
    ("et_ee", "Estonian"),      ("fa_ir", "Persian"),        ("ff_sn", "Fula"),
    ("fi_fi", "Finnish"),       ("fil_ph", "Filipino"),      ("fr_fr", "French"),
    ("ga_ie", "Irish"),         ("gl_es", "Galician"),       ("gu_in", "Gujarati"),
    ("ha_ng", "Hausa"),         ("he_il", "Hebrew"),         ("hi_in", "Hindi"),
    ("hr_hr", "Croatian"),      ("hu_hu", "Hungarian"),      ("hy_am", "Armenian"),
    ("id_id", "Indonesian"),    ("ig_ng", "Igbo"),           ("is_is", "Icelandic"),
    ("it_it", "Italian"),       ("ja_jp", "Japanese"),       ("jv_id", "Javanese"),
    ("ka_ge", "Georgian"),      ("kam_ke", "Kamba"),         ("kea_cv", "Kabuverdianu"),
    ("kk_kz", "Kazakh"),        ("km_kh", "Khmer"),          ("kn_in", "Kannada"),
    ("ko_kr", "Korean"),        ("ky_kg", "Kyrgyz"),         ("lb_lu", "Luxembourgish"),
    ("lg_ug", "Ganda"),         ("ln_cd", "Lingala"),        ("lo_la", "Lao"),
    ("lt_lt", "Lithuanian"),    ("luo_ke", "Luo"),           ("lv_lv", "Latvian"),
    ("mi_nz", "Maori"),         ("mk_mk", "Macedonian"),     ("ml_in", "Malayalam"),
    ("mn_mn", "Mongolian"),     ("mr_in", "Marathi"),        ("ms_my", "Malay"),
    ("mt_mt", "Maltese"),       ("my_mm", "Burmese"),        ("nb_no", "Norwegian"),
    ("ne_np", "Nepali"),        ("nl_nl", "Dutch"),          ("nso_za", "Northern Sotho"),
    ("ny_mw", "Chichewa"),      ("oc_fr", "Occitan"),        ("om_et", "Oromo"),
    ("pa_in", "Punjabi"),       ("pl_pl", "Polish"),         ("ps_af", "Pashto"),
    ("pt_br", "Portuguese"),    ("ro_ro", "Romanian"),       ("ru_ru", "Russian"),
    ("sd_in", "Sindhi"),        ("sk_sk", "Slovak"),         ("sl_si", "Slovenian"),
    ("sn_zw", "Shona"),         ("so_so", "Somali"),         ("sr_rs", "Serbian"),
    ("sv_se", "Swedish"),       ("sw_ke", "Swahili"),        ("ta_in", "Tamil"),
    ("te_in", "Telugu"),        ("tg_tj", "Tajik"),          ("th_th", "Thai"),
    ("tr_tr", "Turkish"),       ("uk_ua", "Ukrainian"),      ("umb_ao", "Umbundu"),
    ("ur_pk", "Urdu"),          ("uz_uz", "Uzbek"),          ("vi_vn", "Vietnamese"),
    ("wo_sn", "Wolof"),         ("xh_za", "Xhosa"),          ("yo_ng", "Yoruba"),
    ("yue_hant_hk", "Cantonese"), ("zu_za", "Zulu"),
]


def extract_mfcc(audio_array, sample_rate=16000, n_mfcc=40):
    mfcc = librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)

    delta = DeltaFeatures()
    mfcc_delta = delta.compute_delta(mfcc)
    mfcc_delta_mean = np.mean(mfcc_delta, axis=1)

    mfcc_delta2 = delta.compute_delta(mfcc_delta)
    mfcc_delta2_mean = np.mean(mfcc_delta2, axis=1)

    return np.concatenate((mfcc_mean, mfcc_delta_mean, mfcc_delta2_mean), axis=0)


def build_features(dataset, label, sample_rate=16000):
    X, y = [], []
    expected_length = 3 * 40

    for i, item in enumerate(dataset):
        audio = np.array(item["audio"]["array"], dtype=np.float32)
        sr = item["audio"]["sampling_rate"]

        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

        audio = quantize(audio , scheme = 2, bit_depth = 2)

        try:
            feat = extract_mfcc(audio)
            if len(feat) != expected_length:
                print(f"  Sample {i}: unexpected feature length {len(feat)}, skipping.")
                continue
            X.append(feat)
            y.append(label)
        except Exception as e:
            print(f"  Sample {i}: error {e}, skipping.")
            continue

    return X, y


# ── Load + extract features for all 102 languages ──────────────────────────
X_all, y_all = [], []
loaded_names = []  # tracks languages that loaded successfully

for label_idx, (lang_code, lang_name) in enumerate(FLEURS_LANGUAGES):
    print(f"[{label_idx+1}/{len(FLEURS_LANGUAGES)}] Loading {lang_name} ({lang_code})...")
    try:
        dataset = load_dataset(
            "google/fleurs", lang_code,
            split="train[:100]",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"  Could not load {lang_code}: {e}. Skipping.")
        continue

    X, y = build_features(dataset, label=label_idx)
    X_all.extend(X)
    y_all.extend(y)
    loaded_names.append(lang_name)
    print(f"  {len(X)} samples added.")

print(f"\nTotal samples: {len(X_all)} across {len(loaded_names)} languages.")

X = np.array(X_all)
y = np.array(y_all)

# ── Train / test split + scaling ───────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── SVM ────────────────────────────────────────────────────────────────────
clf = SVC(kernel="rbf", C=10, gamma="scale")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\n" + classification_report(y_test, y_pred, target_names=loaded_names))

# ── Confusion matrix ───────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(20, 20))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=loaded_names)
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=90)
plt.title("SVM Confusion Matrix — 102 Languages (FLEURS)")
plt.tight_layout()
plt.savefig("confusion_matrix_102.png", dpi=150)
plt.show()