import os

root_pd = r"F:\NADI-Q data\PD_OPEN"
print("Root PD_OPEN:")
for root, dirs, files in os.walk(root_pd):
    print(root, "dirs:", dirs, "files:", files)

print("\nADNI DATA\\ADNIMERGE2\\ADNIMERGE2:")
adni_m = r"F:\NADI-Q data\ADNI DATA\ADNIMERGE2\ADNIMERGE2"
if os.path.exists(adni_m):
    for it in os.listdir(adni_m):
        print(" ", it)

