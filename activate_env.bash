wget http://www.atarimania.com/roms/Roms.rar
unrar e Roms.rar && rm Roms.rar
unzip ROMS.zip && rm *.zip
python3 -m atari_py.import_roms ./ROMS
