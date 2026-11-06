# Billig jämförelse: SiT-S/2, REPA och REG

Den separata ingången `small_experiment.py` kör CIFAR-10 i 64×64 på **en enhet**.
Originalets `train.py` och `generate.py` används inte för dessa experiment.
SiT-S/2 har 12 lager, bredd 384 och patchstorlek 2 på VAE-latenterna.

| Metod | Patch-alignment | Extra CLS-token och CLS-denoising |
|---|---|---|
| sit | Nej | Nej |
| repa | Ja | Nej |
| reg | Ja, samt CLS-alignment | Ja |

Detta är en anpassad metodjämförelse, inte en exakt reproduktion av artikelns
ImageNet-resultat. DINOv2-B får 224×224 bilder; dess 16×16 patch-grid poolas till
4×4 för att matcha SiT:s 16 bildtokens. Alignment djup 4, λ=0.5 och β=0.03
är startvärden från artikelns B-modell, inte validerade optimala värden för S.
Fasta bilder utan slumpmässig augmentation gör att teacher-features kan cachas.
VAE:n och DINOv2 är frysta. Inga DINO-forwardpass görs under träningen.

## 1. Skapa miljö

Kör från repots rot, helst på en dator med NVIDIA-GPU. Python 3.10/3.11 rekommenderas.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -r requirements-small.txt
python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

Installera en PyTorch-build som stöder GPU:n och dess CUDA-miljö enligt
<https://pytorch.org/get-started/locally/> om standardinstallationen inte ger CUDA.
Använd inte originalets `requirements.txt` för denna ingång; den innehåller en
miljöexport med många onödiga beroenden och är inte ett vanligt pip-format.

## 2. Verifiera installationen

```bash
python -m unittest discover -s tests -v
```

Testerna använder syntetiska data och behöver inga nedladdade modellvikter.
De verifierar alla metoders forward/backward, sampling, S-modellens dimensioner
samt att checkpoint + återupptagning ger samma CPU-resultat som oavbruten träning.

## 3. Förbered data en gång

```bash
python small_experiment.py prepare \
  --data data/cifar10-64 \
  --batch-size 32 \
  --device cuda
```

Första körningen hämtar CIFAR-10, officiell DINOv2-kod/vikter via torch.hub och
`stabilityai/sd-vae-ft-mse` via Hugging Face. Internet krävs. Cachen innehåller
alla 50 000 träningsbildernas VAE-medelvärden/standardavvikelser, poolade
DINO-patch-features, CLS-features och etiketter. Räkna med ungefär 1.4 GB för
cachen, utöver nedladdningar och checkpoints. Metadata skrivs först när allt är klart.
Använd en ny katalog om en förberedelse avbrutits.

Öppna `data/cifar10-64/vae_reconstruction.png`: översta raden är bilderna,
nedersta raden VAE-rekonstruktionerna. Kontrollera att informationen bevaras
innan ni lägger tid på träning. VAE-kvaliteten begränsar alla tre metoderna.

För ett billigt tekniskt prov kan ni först skapa en separat cache med
`--limit 128 --data data/cifar10-64-smoke`. Använd hela träningssetet för jämförelsen.

## 4. Kör en pilot för varje metod

```bash
for method in sit repa reg; do
  python small_experiment.py train \
    --method "$method" \
    --data data/cifar10-64 \
    --output "runs/cifar64-$method-seed0" \
    --steps 1000 \
    --batch-size 32 \
    --accumulation 4 \
    --precision fp16 \
    --device cuda \
    --seed 0
done
```

Det ger effektiv batchstorlek 128. Vid minnesbrist kan ni exempelvis använda
`--batch-size 16 --accumulation 8` för **alla tre** metoderna. BF16 kan väljas på
GPU:er som stöder det. På Mac/CPU: använd `--device mps` respektive `cpu` och
`--precision no`. GPU-träning rekommenderas för större körningar.

Data samplas likformigt med återläggning. Antalet optimizersteg är budgeten,
inte antalet epoker. Samma seed ger samma bildindex och samma image-noise/tider
mellan varianterna; olika arkitekturer har inte identiska initiala vikter.
Label dropout är 0.1 även om sampling görs utan CFG.

Varje körning skriver `config.json`, `metrics.jsonl` och `last.pt`.
Checkpointen innehåller modell, EMA, optimizer, AMP-scaler och RNG-tillstånd.
Cacheinnehållet hash-kontrolleras vid start/resume. `last.pt` skrivs över atomärt
vid varje sparintervall och vid sista steget; kopiera den om ni vill behålla
flera träningspunkter. Jämför den uppmätta tiden per steg efter uppvärmning.
Förberedelsens tid ska också redovisas om ni jämför total beräkningskostnad.

## 5. Generera utan classifier-free guidance

```bash
for method in sit repa reg; do
  python small_experiment.py sample \
    --checkpoint "runs/cifar64-$method-seed0/last.pt" \
    --output "samples/cifar64-$method-step1000" \
    --count 100 \
    --batch-size 20 \
    --steps 250 \
    --device cuda \
    --seed 0
done
```

Denna sampler använder alltid **utan CFG (scale=1)**, klassvillkorning,
EMA-vikter och linjär interpolant med SDE Euler–Maruyama, inklusive ett sista
deterministiskt steg. REG genererar sin CLS-latent gemensamt med bildlatenten;
ingen teacher behövs vid sampling. Inställningar läses från checkpointen.
Varje utmatningskatalog måste vara ny. PNG-bilder, `preview.png`, `labels.json`
och `sampling.json` sparas. Klasserna väljs cykliskt 0–9.

100 bilder är en visuell kontroll, inte en statistiskt tillförlitlig FID-mätning.
Använd samma antal genererade bilder, samplingsteg och referensdataset i en
slutlig utvärdering. Använd inte artikelns ImageNet-referensstatistik för CIFAR-10.
Undanta `preview.png` om en metrikkod läser hela katalogen. Någon automatisk
FID-utvärdering ingår inte här.

## 6. Fortsätt träningen om piloten fungerar

```bash
for method in sit repa reg; do
  python small_experiment.py train \
    --method "$method" \
    --data data/cifar10-64 \
    --output "runs/cifar64-$method-seed0" \
    --resume "runs/cifar64-$method-seed0/last.pt" \
    --steps 20000 \
    --batch-size 32 \
    --accumulation 4 \
    --precision fp16 \
    --device cuda \
    --seed 0
done
```

`--steps 20000` betyder 20 000 totalt, inklusive pilotens 1 000 steg.
20 000 är en startbudget och garanterar inte konvergens. Alla inställningar utom
stegbudget/loggning/sparintervall måste matcha när träning återupptas.
Exakt numerisk reproducerbarhet mellan olika enheter/PyTorch-versioner utlovas inte.
Lägg till fler seeds (nya körningskataloger) när upplägget fungerar; en enda seed
räcker inte för starka slutsatser om metodernas skillnader.
