# Sistema de Detección de Objetos Peligrosos en Tiempo Real
# Para Estadios en Ecuador
# Detecta: Botellas, Cuchillos, Bates de baseball, Tijeras

import cv2
import time
from datetime import datetime
from ultralytics import YOLO
import numpy as np

# Configuración de objetos peligrosos
OBJETOS_PELIGROSOS = {
    'bottle': '🍾 BOTELLA',
    'knife': '🔪 CUCHILLO',
    'baseball bat': '⚾ BATE',
    'scissors': '✂️ TIJERAS',
    'fork': '🍴 TENEDOR',
    'spoon': '🥄 CUCHARA'
}

print("="*70)
print("🎥 SISTEMA DE SEGURIDAD PARA ESTADIOS - ECUADOR")
print("="*70)

# Cargar modelo YOLOv8
print("📦 Cargando modelo YOLOv8...")
modelo = YOLO('yolov8n.pt')
print("✅ Modelo cargado correctamente\n")

# Obtener IDs de clases peligrosas
clases_peligrosas_ids = []
for idx, nombre in modelo.names.items():
    if nombre in OBJETOS_PELIGROSOS.keys():
        clases_peligrosas_ids.append(idx)
        print(f"✓ Detectando: {OBJETOS_PELIGROSOS[nombre]}")

print("\n" + "="*70)
print("⌨️ CONTROLES:")
print("   'q' → Salir del sistema")
print("   's' → Guardar captura de pantalla")
print("   'r' → Resetear contador de alertas")
print("="*70 + "\n")

# Iniciar cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: No se pudo acceder a la cámara")
    print("\n💡 Soluciones:")
    print("   1. Verifica que tu webcam esté conectada")
    print("   2. Cierra otras aplicaciones que usen la cámara (Zoom, Teams, etc.)")
    print("   3. Verifica los permisos de la cámara en Windows")
    exit()

# Configurar resolución
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("✅ Cámara activada - Iniciando detección...\n")

# Contadores
frame_count = 0
fps_start_time = time.time()
fps = 0
alertas_totales = 0
log_eventos = []

try:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Error al capturar frame")
            break
        
        # Calcular FPS
        frame_count += 1
        if frame_count >= 30:
            fps = frame_count / (time.time() - fps_start_time)
            fps_start_time = time.time()
            frame_count = 0
        
        # Realizar detección con YOLOv8
        resultados = modelo.predict(source=frame, conf=0.3, verbose=False)
        
        # Procesar resultados
        objetos_detectados = []
        
        for r in resultados:
            # Dibujar las detecciones en el frame
            frame_anotado = r.plot()
            
            # Revisar si hay objetos peligrosos
            for box in r.boxes:
                clase_id = int(box.cls)
                
                # Si es un objeto peligroso
                if clase_id in clases_peligrosas_ids:
                    nombre_clase = modelo.names[clase_id]
                    confianza = float(box.conf)
                    alerta = OBJETOS_PELIGROSOS[nombre_clase]
                    
                    objetos_detectados.append({
                        'nombre': nombre_clase,
                        'alerta': alerta,
                        'confianza': confianza,
                        'bbox': box.xyxy[0].cpu().numpy()
                    })
        
        # Si hay objetos peligrosos, mostrar alertas
        if objetos_detectados:
            alertas_totales += len(objetos_detectados)
            
            # Alerta principal en la parte superior
            cv2.rectangle(frame_anotado, (40, 20), (1240, 80), (0, 0, 255), -1)
            cv2.putText(frame_anotado, 
                       f"ALERTA: {len(objetos_detectados)} OBJETO(S) PELIGROSO(S) DETECTADO(S)",
                       (50, 60),
                       cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
            
            # Listar objetos detectados
            y_pos = 120
            for obj in objetos_detectados:
                # Fondo del texto
                texto = f"{obj['alerta']} - {obj['confianza']:.0%}"
                text_size = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(frame_anotado, (40, y_pos-30), (40+text_size[0]+20, y_pos+10), (0, 100, 255), -1)
                
                cv2.putText(frame_anotado, texto, (50, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                y_pos += 50
                
                # Log en consola
                timestamp = datetime.now().strftime('%H:%M:%S')
                evento = f"[{timestamp}] {obj['alerta']} - Confianza: {obj['confianza']:.0%}"
                if evento not in [e for e in log_eventos[-5:]]:  # Evitar spam
                    print(f"🚨 {evento}")
                    log_eventos.append(evento)
        
        # Información en pantalla
        h, w = frame_anotado.shape[:2]
        
        # FPS
        cv2.putText(frame_anotado, f"FPS: {fps:.1f}", (10, h - 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Estado
        if objetos_detectados:
            estado = "🔴 ALERTA ACTIVA"
            color_estado = (0, 0, 255)
        else:
            estado = "🟢 ZONA SEGURA"
            color_estado = (0, 255, 0)
        
        cv2.putText(frame_anotado, estado, (10, h - 40),
                   cv2.FONT_HERSHEY_DUPLEX, 0.9, color_estado, 2)
        
        # Total de alertas
        cv2.putText(frame_anotado, f"Total alertas: {alertas_totales}", (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Mostrar frame
        cv2.imshow('Sistema de Seguridad - Estadios Ecuador', frame_anotado)
        
        # Controles de teclado
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n🛑 Deteniendo sistema de detección...")
            break
        elif key == ord('s'):
            # Guardar captura
            nombre_archivo = f"captura_alerta_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(nombre_archivo, frame_anotado)
            print(f"📸 Captura guardada: {nombre_archivo}")
        elif key == ord('r'):
            # Resetear contador
            alertas_totales = 0
            log_eventos = []
            print("🔄 Contador de alertas reseteado")

except KeyboardInterrupt:
    print("\n⚠️ Interrupción por teclado detectada")

finally:
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    
    # Resumen final
    print("\n" + "="*70)
    print("📊 RESUMEN DE LA SESIÓN")
    print("="*70)
    print(f"Total de alertas generadas: {alertas_totales}")
    print(f"Eventos únicos registrados: {len(log_eventos)}")
    
    if log_eventos:
        print("\n📋 Últimos 10 eventos:")
        for evento in log_eventos[-10:]:
            print(f"  {evento}")
    
    print("\n✅ Sistema detenido correctamente")
    print("="*70)
