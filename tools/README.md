# tools/

Third-party command-line tools used by the evidence fetcher. Binaries are **not** committed; install them here.

## 7-Zip (`tools/7zip/7z.exe`)

`src/utils/bundle.py` unpacks 7z, RAR, split ZIP (`.z01`), Deflate64 and encrypted ZIP, bz2/xz/gz, VMDK/VHD(X)
and DMG containers with 7-Zip (selective extraction with `-i@listfile`). Tested with 7-Zip 24.09 (x64).

Portable install without admin rights (administrative MSI extraction), from PowerShell in `forensic_agent/`:

```powershell
Invoke-WebRequest -UseBasicParsing https://www.7-zip.org/a/7z2409-x64.msi -OutFile $env:TEMP\7z.msi
Start-Process msiexec.exe -Wait -ArgumentList "/a `"$env:TEMP\7z.msi`" /qn TARGETDIR=`"$PWD\tools\7zip_msi`""
Move-Item tools\7zip_msi\Files\7-Zip tools\7zip; Remove-Item -Recurse tools\7zip_msi
```

Without it, ZIP / TAR / TGZ evidence still works (pure Python); other containers fail with a clear error.
7-Zip is licensed under the GNU LGPL (with the unRAR restriction); see `tools/7zip/License.txt` after install.
