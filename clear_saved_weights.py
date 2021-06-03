from audiomidi import params

for f in params.weights_dir.glob('weights*.hdf5'):
    try:
        f.unlink()
    except:
        print('Cannot delete', f.name)
