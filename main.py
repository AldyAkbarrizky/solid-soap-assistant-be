# main.py
# Backend final: AssemblyAI SDK untuk transkripsi, Gemini untuk diarisasi & S.O.A.P.

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import json
import os
import tempfile
from google import genai
import assemblyai as aai # Menggunakan AssemblyAI SDK

# Inisialisasi aplikasi Flask
app = Flask(__name__)
# Mengizinkan request dari frontend React Anda (sesuaikan port jika berbeda)
# Untuk produksi, Anda bisa mengganti '*' dengan URL frontend Anda
CORS(app, resources={r"/*": {"origins": "*"}})

# --- KONFIGURASI API ---

# PENTING: Atur API Key Anda sebagai environment variable.
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Konfigurasi AssemblyAI SDK
if ASSEMBLYAI_API_KEY:
    aai.settings.api_key = ASSEMBLYAI_API_KEY
    print("AssemblyAI API Key berhasil dikonfigurasi.")
else:
    print("Peringatan: Environment variable ASSEMBLYAI_API_KEY tidak ditemukan.")

# Konfigurasi Gemini
try:
    gemini_client = genai.Client(api_key = GEMINI_API_KEY)
    gemini_model_name = "models/gemini-2.0-flash"
    print(f"Koneksi ke Google Gemini API berhasil.")
except Exception as e:
    print(f"Gagal mengkonfigurasi Gemini API: {e}")
    gemini_client = None

# --- FUNGSI-FUNGSI AI ---

def transcribe_with_assemblyai_sdk(audio_path):
    """
    Menggunakan AssemblyAI Python SDK untuk mendapatkan transkrip mentah.
    """
    print("Memulai transkripsi dengan AssemblyAI SDK...")
    
    config = aai.TranscriptionConfig(language_code="id")
    
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_path, config)

    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f"Transkripsi AssemblyAI gagal: {transcript.error}")

    print("Transkripsi AssemblyAI SDK selesai.")
    return transcript.text or ""

def call_gemini_for_diarization(raw_transcript):
    """
    Memanggil Gemini untuk memperbaiki typo dan melakukan diarisasi.
    """
    print("Memulai perbaikan transkrip dan diarisasi dengan Gemini...")
    
    prompt = f"""Tugas Anda adalah sebagai editor transkrip medis profesional.
Anda menerima sebuah teks mentah hasil transkripsi AI dari percakapan dokter dan pasien.
Lakukan dua hal:
1.  Perbaiki semua kesalahan ketik (typo), ejaan, dan tata bahasa agar menjadi Bahasa Indonesia yang baik dan benar. Perhatikan untuk nama nama ilmiah (nama penyakit atau nama obat), beserta takarannya! Tidak boleh SALAH, pastikan nama obatnya benar! Jika teks dari transcriptnya salah, perbaiki menjadi nama ilmiah yangs sesuai!
2.  Format ulang teks yang sudah diperbaiki menjadi dialog yang jelas antara "Dokter" dan "Pasien". Pastikan Anda mengatribusikan ucapan ke pembicara yang tepat secara logis.

Setiap baris harus diawali dengan "Dokter: " atau "Pasien: ".

---
**Transkrip Mentah:**
{raw_transcript}
---
**Hasil Dialog yang Sudah Diperbaiki dan Diformat:**
"""
    
    if not gemini_client:
        raise ConnectionError("Client Gemini tidak terinisialisasi.")
        
    response = gemini_client.models.generate_content(
        model=gemini_model_name,
        contents=prompt
    )
    diarized_text = response.text.strip()
    print("Diarisasi dan perbaikan selesai.")
    return diarized_text

def get_soap_prompt(diarized_transcript):
    """Membuat prompt untuk S.O.A.P."""
    return f"""Tugas Anda adalah bertindak sebagai asisten medis AI.
Berdasarkan dialog berikut, buat ringkasan medis dalam format S.O.A.P.
Gunakan format teks biasa (plain text). JANGAN gunakan Markdown (`**`).

Contoh Format:
S.O.A.P. - CATATAN MEDIS

SUBJECTIVE (Keluhan Subjektif):
- Pasien mengeluh pusing.
OBJECTIVE (Temuan Objektif):
- Suhu: 37.8°C.
ASSESSMENT (Penilaian):
- Demam e.c. infeksi virus.
PLAN (Rencana Tindakan):
- Pemberian paracetamol.
DIAGNOSIS (Kesimpulan Diagnosa):
- Infeksi Saluran Pernapasan Atas (ISPA)
ICD-10 (Kode ICD-10):
- J00

---
**Dialog Percakapan:**
{diarized_transcript}
---
**Hasil S.O.A.P.:**
"""

def stream_gemini_response(prompt):
    """Generator untuk streaming respons dari Gemini."""
    print("Mulai streaming respons dari Gemini...")
    try:
        stream = gemini_client.models.generate_content(
            model=gemini_model_name, contents=prompt, stream=True
        )
        for chunk in stream:
            cleaned_text = chunk.text.replace('**', '')
            yield cleaned_text
    except Exception as e:
        yield f"Error streaming dari Gemini: {e}"


# --- ENDPOINTS API ---
@app.route('/process-audio', methods=['POST'])
def process_audio():
    """Endpoint untuk proses lengkap: AssemblyAI -> Diarisasi Gemini -> S.O.A.P. Gemini."""
    if gemini_client is None or not ASSEMBLYAI_API_KEY:
        return jsonify({"error": "API Key tidak berhasil dikonfigurasi di server."}), 500

    if 'audio' not in request.files:
        return jsonify({"error": "Tidak ada file audio yang dikirim"}), 400

    audio_file = request.files['audio']
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, audio_file.filename)
    audio_file.save(temp_path)

    try:
        raw_transcript_text = transcribe_with_assemblyai_sdk(temp_path)
        
        output_filename_raw = "raw_transcript_result.txt"
        with open(output_filename_raw, 'w', encoding='utf-8') as f:
            f.write(raw_transcript_text)
        print(f"Hasil transkrip mentah dari AssemblyAI telah disimpan di: {output_filename_raw}")
        
        print("\n--- HASIL TRANSKRIP MENTAH ASSEMBLYAI ---\n")
        print(raw_transcript_text)
        print("\n----------------------------------------\n")
        
        diarized_transcript = call_gemini_for_diarization(raw_transcript_text)
        
        soap_prompt = get_soap_prompt(diarized_transcript)
        soap_response = gemini_client.models.generate_content(
            model=gemini_model_name, contents=soap_prompt
        )
        soap_content = soap_response.text.replace('**', '').strip()
        
        return jsonify({
            "transcript": diarized_transcript,
            "soapContent": soap_content
        })
    except Exception as e:
        print(f"Error selama proses: {e}")
        return jsonify({"error": f"Gagal memproses: {e}"}), 500
    finally:
        os.remove(temp_path)

@app.route('/regenerate-soap-stream', methods=['POST'])
def regenerate_soap_stream():
    """Endpoint untuk membuat ulang S.O.A.P. dengan STREAMING dari Gemini."""
    data = request.get_json()
    if not data or 'transcript' not in data:
        return jsonify({"error": "Transkrip tidak ditemukan dalam request"}), 400
    
    diarized_transcript = data['transcript']
    soap_prompt = get_soap_prompt(diarized_transcript)
    
    return Response(stream_with_context(stream_gemini_response(soap_prompt)), mimetype='text/plain')

# Hapus blok 'if __name__ == '__main__':' karena Render akan menjalankannya secara berbeda
