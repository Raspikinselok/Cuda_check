import torch

def check_cuda():
    # Überprüfen, ob CUDA verfügbar ist
    cuda_available = torch.cuda.is_available()
    print(f"CUDA verfügbar: {cuda_available}")

    if cuda_available:
        # Anzahl und Name der CUDA-Geräte
        num_devices = torch.cuda.device_count()
        print(f"Anzahl verfügbarer CUDA-Geräte: {num_devices}")
        
        for i in range(num_devices):
            print(f"Gerät {i}: {torch.cuda.get_device_name(i)}")
            
        # Einfache Tensor-Operationen mit CUDA
        tensor = torch.rand((1000, 1000))
        
        # Verschieben des Tensors auf das erste CUDA-Gerät, wenn verfügbar
        device = torch.device("cuda:0")  # Ändere den Index, wenn du ein anderes Gerät verwenden möchtest
        tensor = tensor.to(device)
        
        # Einfache Berechnung
        result = tensor * tensor
        print("Berechnung auf dem CUDA-Gerät erfolgreich!")
    else:
        print("Keine CUDA-Geräte verfügbar.")

# Führen die Überprüfung aus
check_cuda()
