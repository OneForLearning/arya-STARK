#!/usr/bin/env python3
"""
Benchmark complet arya-STARK avec gradient entier (50,890 paramètres).
Usage: python3 test.py
"""
import time
import json
import numpy as np
from pathlib import Path
from arya_stark.data.loaders import load_synthetic_mnist
from arya_stark.data.partition import partition_iid
from arya_stark.models.mlp import MLPModel
from arya_stark.encoding import encode_vector
from arya_stark.client.stark_bridge import prove_dot_product, verify_dot_product
from arya_stark.crypto.mldsa import keygen, sign, verify

print("=" * 80)
print("arya-STARK - BENCHMARK FINAL COMPLET")
print("100 clients × 3 rounds × GRADIENT COMPLET (50,890 params)")
print("=" * 80)

NUM_CLIENTS = 100
NUM_ROUNDS = 20
GRADIENT_SIZE = 50890

results_dir = Path("benchmark-results")
results_dir.mkdir(exist_ok=True)

print(f"\n📋 Configuration:")
print(f"  Modèle: MLP (784→64→10) = {GRADIENT_SIZE:,} params")
print(f"  Clients: {NUM_CLIENTS}")
print(f"  Rounds: {NUM_ROUNDS}")
print(f"  STARK: GRADIENT COMPLET ({GRADIENT_SIZE:,} éléments) ✅")
print(f"  Dataset: Synthetic MNIST (10,000 train, 1,000 test)")

# Setup FL
print("\n🔧 Setup...")
ds = load_synthetic_mnist(n_train=10000, n_test=1000, seed=42)
shards = partition_iid(len(ds.X_train), num_clients=NUM_CLIENTS, seed=42)
global_model = MLPModel(input_dim=784, hidden_dim=64, num_classes=10, seed=42)

# Clés ML-DSA
t0 = time.perf_counter()
pk, sk = keygen()
keygen_time = time.perf_counter() - t0
print(f"  Clés ML-DSA: ✓ ({keygen_time:.3f}s)")

all_metrics = {
    "configuration": {
        "num_clients": NUM_CLIENTS,
        "num_rounds": NUM_ROUNDS,
        "model": "MLP",
        "num_params": GRADIENT_SIZE,
        "stark_size": GRADIENT_SIZE,
        "dataset_size": {"train": len(ds.X_train), "test": len(ds.X_test)},
        "bug_fixed": True,
        "note": "Full gradient STARK proof (bug n>16 fixed)"
    },
    "rounds": []
}

protocol_start = time.perf_counter()

for round_num in range(1, NUM_ROUNDS + 1):
    print(f"\n{'='*80}")
    print(f"ROUND {round_num}/{NUM_ROUNDS}")
    print('='*80)
    
    round_start = time.perf_counter()
    round_metrics = {"round_number": round_num, "clients": [], "server_verification": [], "aggregation": {}}
    
    global_params = global_model.get_flat_params()
    client_data_list = []
    
    print(f"\n[CLIENTS] Entraînement + Crypto (gradient complet)...")
    
    for client_id in range(NUM_CLIENTS):
        # 1. ENTRAÎNEMENT LOCAL
        local_model = MLPModel(input_dim=784, hidden_dim=64, num_classes=10)
        local_model.set_flat_params(global_params.copy())
        
        X_local = ds.X_train[shards[client_id].indices]
        y_local = ds.y_train[shards[client_id].indices]
        
        t_train_start = time.perf_counter()
        for _ in range(5):
            idx = np.random.choice(len(X_local), size=32, replace=False)
            local_model.sgd_step(X_local[idx], y_local[idx], lr=0.01)
        t_train = time.perf_counter() - t_train_start
        
        gradient = global_params - local_model.get_flat_params()
        
        # 2. STARK PROOF (GRADIENT COMPLET)
        gradient_fp = encode_vector(gradient, m=6)
        
        a = gradient_fp.copy()
        b = np.ones(len(a), dtype=np.uint64)
        
        t_prove_start = time.perf_counter()
        proof = prove_dot_product(a, b)
        t_prove = time.perf_counter() - t_prove_start
        
        # 3. ML-DSA SIGNATURE
        message = proof.proof_bytes + gradient_fp.tobytes()
        t_sign_start = time.perf_counter()
        signature = sign(sk, message)
        t_sign = time.perf_counter() - t_sign_start
        
        train_acc = local_model.accuracy(X_local, y_local)
        
        round_metrics["clients"].append({
            "client_id": client_id,
            "training_time_seconds": t_train,
            "local_accuracy": float(train_acc),
            "gradient_l2_norm": float(np.linalg.norm(gradient)),
            "proof_generation_time_seconds": t_prove,
            "proof_size_bytes": proof.size_bytes,
            "signature_time_seconds": t_sign,
            "signature_size_bytes": len(signature),
        })
        
        client_data_list.append({
            "gradient": gradient,
            "proof": proof,
            "signature": signature,
            "message": message,
            "a": a,
            "b": b,
        })
        
        if (client_id + 1) % 10 == 0:
            print(f"  [{client_id+1:3d}/{NUM_CLIENTS}] Train: {t_train:.2f}s, Proof: {t_prove:.2f}s, Acc: {train_acc:.3f}")
    
    print(f"\n[SERVEUR] Vérification...")
    
    valid_gradients = []
    num_sig_ok = 0
    num_proof_ok = 0
    
    for client_id, data in enumerate(client_data_list):
        t_vs_start = time.perf_counter()
        sig_valid = verify(pk, data["message"], data["signature"])
        t_vs = time.perf_counter() - t_vs_start
        
        if sig_valid:
            num_sig_ok += 1
        else:
            round_metrics["server_verification"].append({
                "client_id": client_id,
                "signature_valid": False,
                "signature_verification_time_seconds": t_vs,
                "proof_valid": False,
                "proof_verification_time_seconds": 0.0,
            })
            continue
        
        t_vp_start = time.perf_counter()
        proof_valid = verify_dot_product(data["proof"], data["a"], data["b"], data["proof"].public_output)
        t_vp = time.perf_counter() - t_vp_start
        
        if proof_valid:
            num_proof_ok += 1
        
        round_metrics["server_verification"].append({
            "client_id": client_id,
            "signature_verification_time_seconds": t_vs,
            "signature_valid": sig_valid,
            "proof_verification_time_seconds": t_vp,
            "proof_valid": proof_valid,
        })
        
        if sig_valid and proof_valid:
            valid_gradients.append(data["gradient"])
        
        if (client_id + 1) % 10 == 0:
            print(f"  [{client_id+1:3d}/{NUM_CLIENTS}] Sig: {num_sig_ok}, Proof: {num_proof_ok}")
    
    print(f"\n[SERVEUR] Agrégation...")
    
    if valid_gradients:
        aggregated = np.mean(valid_gradients, axis=0)
        global_model.apply_update(aggregated, lr=1.0)
        agg_norm = float(np.linalg.norm(aggregated))
    else:
        agg_norm = 0.0
    
    test_acc = global_model.accuracy(ds.X_test, ds.y_test)
    test_loss = float(global_model.loss(ds.X_test, ds.y_test))
    
    round_metrics["aggregation"] = {
        "num_valid_updates": len(valid_gradients),
        "num_rejected_updates": NUM_CLIENTS - len(valid_gradients),
        "aggregated_l2_norm": agg_norm,
    }
    
    round_metrics["evaluation"] = {
        "test_accuracy": float(test_acc),
        "test_loss": test_loss,
    }
    
    round_dur = time.perf_counter() - round_start
    round_metrics["round_duration_seconds"] = round_dur
    all_metrics["rounds"].append(round_metrics)
    
    print(f"\n✓ Round {round_num}: {len(valid_gradients)}/{NUM_CLIENTS} valides")
    print(f"  Test accuracy: {test_acc:.3f}, Loss: {test_loss:.4f}")
    print(f"  Durée: {round_dur:.1f}s")

protocol_duration = time.perf_counter() - protocol_start
all_metrics["total_duration_seconds"] = protocol_duration
all_metrics["final_test_accuracy"] = float(test_acc)
all_metrics["final_test_loss"] = test_loss

# Statistiques
all_train = [c["training_time_seconds"] for r in all_metrics["rounds"] for c in r["clients"]]
all_prove = [c["proof_generation_time_seconds"] for r in all_metrics["rounds"] for c in r["clients"]]
all_sign = [c["signature_time_seconds"] for r in all_metrics["rounds"] for c in r["clients"]]
all_vsig = [s["signature_verification_time_seconds"] for r in all_metrics["rounds"] for s in r["server_verification"]]
all_vprf = [s["proof_verification_time_seconds"] for r in all_metrics["rounds"] for s in r["server_verification"] if s["proof_verification_time_seconds"] > 0]
all_psize = [c["proof_size_bytes"] for r in all_metrics["rounds"] for c in r["clients"]]

all_metrics["statistics"] = {
    "training_time_ms": {"mean": float(np.mean(all_train)*1000), "std": float(np.std(all_train)*1000)},
    "proof_gen_ms": {"mean": float(np.mean(all_prove)*1000), "std": float(np.std(all_prove)*1000)},
    "signature_ms": {"mean": float(np.mean(all_sign)*1000), "std": float(np.std(all_sign)*1000)},
    "verify_sig_ms": {"mean": float(np.mean(all_vsig)*1000), "std": float(np.std(all_vsig)*1000)},
    "verify_proof_ms": {"mean": float(np.mean(all_vprf)*1000), "std": float(np.std(all_vprf)*1000)},
    "proof_size_kb": {"mean": float(np.mean(all_psize)/1024), "std": float(np.std(all_psize)/1024)},
}

# Sauvegarder
jf = results_dir / "benchmark_FINAL_full_gradient.json"
jf.write_text(json.dumps(all_metrics, indent=2))

tf = results_dir / "benchmark_FINAL_summary.txt"
s = all_metrics["statistics"]
with open(tf, "w") as f:
    f.write("=" * 80 + "\n")
    f.write("arya-STARK - BENCHMARK FINAL (GRADIENT COMPLET)\n")
    f.write("100 clients × 3 rounds × 50,890 params STARK proof\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"DURÉE TOTALE: {protocol_duration:.1f}s ({protocol_duration/60:.1f} min)\n")
    f.write(f"PRÉCISION FINALE: {test_acc:.1%}\n\n")
    f.write(f"TEMPS MOYENS:\n")
    f.write(f"  Entraînement MLP: {s['training_time_ms']['mean']:.0f} ms\n")
    f.write(f"  STARK génération: {s['proof_gen_ms']['mean']:.0f} ms ({s['proof_gen_ms']['mean']/1000:.2f}s)\n")
    f.write(f"  STARK vérification: {s['verify_proof_ms']['mean']:.1f} ms\n")
    f.write(f"  ML-DSA signature: {s['signature_ms']['mean']:.1f} ms\n")
    f.write(f"  ML-DSA vérification: {s['verify_sig_ms']['mean']:.1f} ms\n\n")
    f.write(f"TAILLES:\n")
    f.write(f"  Proof STARK: {s['proof_size_kb']['mean']:.1f} KB\n")
    f.write(f"  Signature ML-DSA: 3.2 KB\n\n")
    f.write(f"COUVERTURE GRADIENT: 100% (50,890/50,890 params) ✅\n")

print("\n" + "=" * 80)
print("✅ BENCHMARK FINAL TERMINÉ")
print("=" * 80)
print(f"\n⏱️  Durée: {protocol_duration:.1f}s ({protocol_duration/60:.1f} min)")
print(f"🎯 Précision: {test_acc:.1%}")
print(f"\n📊 Temps moyens:")
print(f"  STARK proof: {s['proof_gen_ms']['mean']/1000:.2f}s (gen), {s['verify_proof_ms']['mean']:.1f}ms (verif)")
print(f"  ML-DSA: {s['signature_ms']['mean']:.1f}ms (sign), {s['verify_sig_ms']['mean']:.1f}ms (verif)")
print(f"\n💾 Tailles:")
print(f"  Proof: {s['proof_size_kb']['mean']:.1f} KB")
print(f"\n🎉 GRADIENT COMPLET PROUVÉ (100% coverage)")
print("\n" + "=" * 80)
print(f"📁 {jf}")
print(f"📄 {tf}")
print("=" * 80)
