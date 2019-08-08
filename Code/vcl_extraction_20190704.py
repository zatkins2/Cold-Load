#!/usr/bin/python -u
#author: mlungu 20190206

import os
import sys
import numpy as np

# Where are inputs and outputs?

fridge = raw_input("Which fridge (lowercase institution)?: ")

datadir = '/home/zatkins/repos/data/sopton/oxford/{}/raw/'.format(fridge)
outdir = '/home/zatkins/repos/data/sopton/oxford/{}/extracted/'.format(fridge)

# Run through all available log files

datafns = np.sort([fn for fn in os.listdir(datadir) if '.vcl' in fn])

for fn in datafns:
  datafp = datadir+fn
  outfp = outdir+fn.split('.')[0]

  if not os.path.exists(outfp):
    os.makedirs(outfp)

  # Open file and read header

  datafile = open(datafp,'rb')
  header = datafile.read(6144)

  headerfile = open(outfp+'/header.txt', 'w')

  for byte in header:
    if byte != '\0':
      headerfile.write(byte)

  headerfile.close()

  # Read the field names

  fields = np.array([])

  for i in range(184):
    field = datafile.read(32)

    if field[0] != '\0':
      fields = np.append(fields,field)

  # Read the data

  datafile.read(256)

  data = []

  while True:
    bpl = datafile.read(8)

    if bpl == '':
      break

    bpl = int(np.frombuffer(bpl, np.dtype(float))[0])
    packet = np.zeros(bpl/8)
    packet[0] = bpl

    for i in range(1,bpl/8):
      packet[i] = np.frombuffer(datafile.read(8), np.dtype(float))[0]

    data.append(packet)

  data = np.array(data).T

  datafile.close()

  # Write the data to disk

  flds = np.array([])

  for i in range(len(fields)):

    if 'LineSize' in fields[i] or 'LineNumber' in fields[i]:
      continue

    fname = fields[i].split('(')[0].strip().replace(' ','_')
    fname = fname.replace('[','').replace(']','')+'.p'
    flds = np.append(flds,fields[i])

    try:
      data[i].dump(outfp+'/'+fname)
    except IndexError:
      data.dump(outfp+'/'+fname)

  np.savetxt(outfp+'/fields.txt', flds, fmt = '%s')

  print fn

