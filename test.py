from primerize_degen.primerize_1d import Primerize_1D

worker: Primerize_1D = Primerize_1D()

worker.set("MIN_LENGTH", 10)
worker.set("MAX_LENGTH", 65)
SEQUENCE: str = """AATTTCAACATGNNKAAGAATGACATGGTGGATCAGATGCAGGAGGACGTGATTTCCATCTGGGATCAGTGCCTGAAACCCTGCGTGAAGCTGACCAACACCTCCACACTGACTCAGGCTTGTCCCAAGGTGACATTCGACCCTATTCCAATCCACTACTGCGCTCCTGCAGGCTATGCCATCCTGAAATGTAACAATAAGACCTTTAACGGCAAAGGGCCATGCAACAATGTGAGCACTGTCCAGTGTACCCACGGCATCAAGCCCGTGGTCTCAACACAGCTGCTGCTGAACGGGAGCCTGGCAGAGGAAGAGATTGTGATCAGATCAAAAAACCTGNNKNNKNNKNNKNNKATCATTATCGTGCAGCTGAATAAGAGTGTGGAGATCGTCTGCACACGACCTAACAATGGCGGCAGCGGATCTGGGGGAGATATTCGGCAGGCTTATTGTAACATCAGTGGCAGAAATTGGTCAGAAGCCGTGAACCAGGTCAAGAAAAAGCTGAAAGAGCACTTCCCCCATAAGAATATTAGCTTTCAGTCTAGTTCAGGCGGGGACCTGGAAATCACCACACACTCCTTCAACTGCGGAGGCGAGTTCTTTTACTGTAATACATCCGGCCTGTTTAACGATACCATTTCTAATGCCACAATCATGCTGCCTTGCCGGATCAAGCAGATTATCAACATGTGGCAGGAAGTGGGAAAGGCTATCTATGCACCACCCATCAAGGGCAATATCACCTGTAAGAGTGACATTACAGGGCTGCTGCTGNNKAGAGATGGGGGAGACACTACCGATAACACCGAGATTTTCCGGCCTAGCGGAGGAGACATGCGAGATAATTGGCGGTCTGAACTGTACAAATATAAGGTGGTCGAGATCAAGCCTCTGGGATCCGAACAAAAGCTTATTT"""

job = worker.design(sequence=SEQUENCE)


for i, primer in enumerate(job.primer_set, start=1):
    print(f"Test Primer {i},{primer}\n")
