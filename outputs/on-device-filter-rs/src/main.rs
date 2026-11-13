// Spike: prove the sherpa-onnx crate loads the GigaSpeech model and returns token timestamps.
use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig, OfflineTransducerModelConfig};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (model, wav) = (&args[1], &args[2]);
    let mut config = OfflineRecognizerConfig::default();
    let file = |kind: &str| Some(format!("{model}/{kind}-epoch-30-avg-1.int8.onnx"));
    config.model_config.transducer = OfflineTransducerModelConfig {
        encoder: file("encoder"), decoder: file("decoder"), joiner: file("joiner"),
    };
    config.model_config.tokens = Some(format!("{model}/tokens.txt"));
    config.model_config.num_threads = 2;
    let recognizer = OfflineRecognizer::create(&config).expect("create recognizer");
    let mut reader = hound::WavReader::open(wav).expect("open wav");
    let step = reader.spec().sample_rate as usize / 16000;
    let pcm: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
    let mono: Vec<f32> = pcm.chunks(2).step_by(step)
        .map(|f| (f[0] as f32 + f[1] as f32) / 65536.0).collect();
    let start = std::time::Instant::now();
    let stream = recognizer.create_stream();
    stream.accept_waveform(16000, &mono);
    recognizer.decode(&stream);
    let result = stream.get_result().expect("result");
    println!("{:?} {:?} {:?} in {:?}", result.text, result.tokens, result.timestamps, start.elapsed());
}
