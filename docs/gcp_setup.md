# GCP VM Setup for TAFNet

This walkthrough sets up a GCP Compute Engine VM with a GPU, mounts your
Google Drive at `/mnt/drive` via `rclone`, and installs all dependencies needed
to run the TAFNet pipeline end-to-end.

It assumes the ADNI data (DICOM and / or `.nii.gz`) lives on the **same Google
account** that owns the GCP project.


## 1. Create the VM

Pick a machine with a recent NVIDIA GPU. A reasonable starting point:

| Field                 | Value                                      |
|-----------------------|--------------------------------------------|
| Machine type          | `n1-standard-8` (8 vCPU, 30 GB RAM)        |
| GPU                   | 1 × NVIDIA T4 (or L4 if available)         |
| Boot disk OS          | Ubuntu 22.04 LTS                           |
| Boot disk size        | ≥ 200 GB SSD                               |
| Firewall              | Allow SSH only                             |

Either through the Cloud Console or via gcloud:

```bash
gcloud compute instances create tafnet-vm \
    --zone=australia-southeast1-c \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True"
```

The `install-nvidia-driver=True` metadata flag triggers GCP's standard NVIDIA
driver installer the first time the VM boots. Verify on the VM with `nvidia-smi`.


## 2. System packages

SSH into the VM, then:

```bash
sudo apt-get update
sudo apt-get install -y \
    python3.10 python3.10-venv python3-pip \
    git build-essential pkg-config \
    dcm2niix \
    fuse3 ca-certificates curl
```

If `nvidia-smi` does not yet work, install drivers manually (or wait a couple
of minutes after first boot and reboot once).


## 3. CUDA-enabled PyTorch

`requirements.txt` pins `torch>=2.0` but does not select a CUDA build. Install
the right wheel for your CUDA version **before** `pip install -r
requirements.txt`. For CUDA 12.1, for example:

```bash
python3.10 -m venv ~/tafnet-venv
source ~/tafnet-venv/bin/activate
pip install --upgrade pip
pip install torch==2.3.1 torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

Then install everything else inside the same venv:

```bash
cd ~/tafnet
pip install -r requirements.txt
```


## 4. Install rclone

```bash
curl https://rclone.org/install.sh | sudo bash
rclone version    # sanity check
```


## 5. Configure a Google Drive remote

Run `rclone config` and follow the interactive prompts. The values that matter
look like this:

```
n) New remote
name> mydrive
Storage> drive
client_id>           (leave blank to use rclone's shared client)
client_secret>       (leave blank)
scope> 1             (Full access)
service_account_file>(leave blank)
Edit advanced config? n
Use auto config? n   (the VM has no browser)
```

`rclone` will print a URL. Open it on your laptop, sign in with the **same
Google account** that owns the Drive data, and paste the verification code
back into the VM. Confirm the team-drive prompt with `n`.

Verify with:

```bash
rclone lsd mydrive:
```

You should see your top-level Drive folders. If you see an empty list, you
authenticated against the wrong Google account.


## 6. Mount the Drive at /mnt/drive

Pick a mount point and create it:

```bash
sudo mkdir -p /mnt/drive
sudo chown "$USER":"$USER" /mnt/drive
```

Mount it as a daemon (the `--vfs-cache-mode writes` flag is important — ANTs
and `dcm2niix` both write multi-pass files):

```bash
rclone mount mydrive: /mnt/drive \
    --daemon \
    --vfs-cache-mode writes \
    --dir-cache-time 24h \
    --buffer-size 64M \
    --log-file ~/rclone.log \
    --log-level INFO
```

Confirm the mount:

```bash
ls /mnt/drive
mount | grep rclone
```

To unmount later: `fusermount3 -u /mnt/drive`.


## 7. Persist the mount across reboots (optional)

If you reboot the VM often, install a systemd unit so the mount comes up
automatically. Create `/etc/systemd/system/rclone-drive.service`:

```ini
[Unit]
Description=rclone mount Google Drive at /mnt/drive
AssertPathIsDirectory=/mnt/drive
After=network-online.target
Wants=network-online.target

[Service]
Type=notify
User=YOUR_LINUX_USER
Group=YOUR_LINUX_USER
ExecStart=/usr/bin/rclone mount mydrive: /mnt/drive \
    --vfs-cache-mode writes \
    --dir-cache-time 24h \
    --buffer-size 64M
ExecStop=/bin/fusermount3 -u /mnt/drive
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now rclone-drive.service
systemctl status rclone-drive.service
```


## 8. Verify the TAFNet pipeline sees Drive

From the project root, with the venv active:

```bash
python - <<'PY'
import sys; sys.path.insert(0, "src")
from tafnet.utils.drive import verify_drive_mount
verify_drive_mount("/mnt/drive/MyDrive")     # adjust to your actual data dir
PY
```

A successful run prints `[OK] Drive mount looks healthy`. If it complains, the
most common causes are: the mount fell off (re-run the `rclone mount` command
or the systemd unit), or the path inside Drive does not exist (check with
`ls /mnt/drive/MyDrive`).


## 9. Run the pipeline

Edit `configs/preprocessing.yaml` and `configs/default.yaml` to point at the
actual Drive folders for your ADNI dataset, then:

```bash
python scripts/00_dicom_to_nifti.py --config configs/preprocessing.yaml
python scripts/01_preprocess.py     --config configs/preprocessing.yaml
python scripts/02_train.py          --config configs/default.yaml
python scripts/03_evaluate.py       --config configs/default.yaml
```

Training Phase 4 alone takes hours on a single T4; the full Phase 5/6 sweep
across five folds and all benchmarks runs overnight. Use `tmux` or `screen` so
you can disconnect from SSH without killing the job.


## Alternative: use a GCS bucket instead of Drive

If you eventually need higher throughput than Drive can supply, move the data
to a GCS bucket and mount it with `gcsfuse`:

```bash
sudo apt-get install -y gcsfuse
gcsfuse --implicit-dirs YOUR_BUCKET /mnt/drive
```

The TAFNet code itself does not care which FUSE filesystem is behind
`/mnt/drive`; only `verify_drive_mount` will need its name adjusted.


## Cost tips

* Stop the VM (`gcloud compute instances stop tafnet-vm`) whenever you are not
  actively running jobs — GPU minutes are the dominant cost.
* The boot disk costs nothing extra while stopped.
* Detach the GPU and restart as a plain `n1-standard-8` for cheap
  preprocessing / evaluation passes that do not need a GPU.
