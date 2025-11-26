from jiwer import wer

def load_lines(path):
    return [line.strip().split("\t")[1] for line in open(path, encoding="utf8")]

def compute_wer(refs_path, hyp_path):
    ref = load_lines(refs_path)
    hyp = load_lines(hyp_path)
    score = wer(ref, hyp)
    print("WER =", score)

if __name__ == "__main__":
    compute_wer("data/test/references.txt", "finetuned_transcriptions.txt")
