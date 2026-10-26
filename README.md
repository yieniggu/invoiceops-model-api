# InvoiceOps Model API

> **Bootstrap transversal:** siga el
> [runbook central de Stack](../invoiceops-stack/LOCAL_INFRASTRUCTURE_RUNBOOK.md)
> para el orden de un entorno local limpio. Este README es propietario de la
> imagen, configuración, health de modelo y release smoke; un health de proceso
> o una receta aislada no sustituyen la ejecución integrada ni validan remoto
> (`NOT_VERIFIED`).

Servicio HTTP mínimo para exponer modelos de InvoiceOps. Usa un catálogo local
explícito y carga sus modelos desde MLflow durante el inicio del proceso.

**Evidencia local vigente (2026-10-19):** STACK-02 **COMPLETED — VALIDATED**;
ops08 estándar fresco **PASS, 11/11 pruebas**, salida 0 en **556.121 s**.
Incluye startup con champion y health/predict **200/200** con versión/run
coincidentes, release, alias ausente y restauración sobre PostgreSQL/MinIO
habituales. Véase el [cierre y ledger CR006–CR011 de Stack](../invoiceops-stack/README.md#cierre-vigente-stack-02--prueba-estándar-ops08).
No reclasifica API-06 (**ACCEPTED_RISK**) ni prueba Cloud Run, integración App,
fiabilidad universal en frío o preparación productiva. No autoriza operaciones.

**Alcance de esa evidencia:** ops08 corresponde a los bytes anteriores al cambio
de aislamiento de salida CR-012 descrito abajo; no se ha repetido contra este
launcher. CR012 queda **RESOLVED_LOCAL** tras revisión independiente y prueba
Linux acotada; el veredicto del lote es **READY_WITH_FOLLOW_UP**. CR013 de STACK-02
permanece **OPEN — FOLLOW_UP_REQUIRED**, sin corrección autorizada ni aceptación
de riesgo implícita; véase el [ledger vigente](../invoiceops-stack/README.md#cierre-de-hallazgos-stack-02).

## Inicio rápido

Requiere Python 3.12 y [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
export MLFLOW_TRACKING_URI="https://mlflow.example.internal"
export MODEL_API_ADMIN_TOKEN="..."
uv run uvicorn invoiceops_model_api.app:app --host 0.0.0.0 --port 8000
```

En otra terminal, verifica el servicio:

```bash
curl http://127.0.0.1:8000/health
```

La respuesta esperada es:

```json
{ "status": "ok" }
```

## Configuración de MLflow

`MLFLOW_TRACKING_URI` es obligatoria y no puede ser vacía o contener sólo
espacios. Si falta o es inválida, el arranque falla cerrado: el proceso no queda
operacional. El servicio sólo lee la configuración desde variables de entorno
en el arranque; no acepta credenciales por HTTP ni desde archivos de configuración.

La autenticación básica es opcional, pero `MLFLOW_TRACKING_USERNAME` y
`MLFLOW_TRACKING_PASSWORD` forman un par atómico: ambas deben estar definidas o
ambas ausentes. Una configuración parcial también falla cerrado al arrancar. No
incluyas estos valores en el repositorio, imágenes, logs o respuestas HTTP.

**Compatibilidad operativa:** el pin del cliente es `mlflow==3.16.0`. Antes de
atribuir compatibilidad con un servidor 3.16.1, prueba en un Workspace aislado
(sin promover el alias de la instalación operada) un run sintético, su
versión y alias, la carga real por `models:/` y el smoke HTTP
`refresh → health → predict` con versión y run esperados. La coincidencia de
versiones o un test con doble de cliente no sustituyen esta prueba. Si el
servidor no alcanza el estado saludable, reporta bloqueo de entorno; no cambies
el pin ni incrementes timeouts para presentar una compatibilidad no acreditada.

La prueba host-local con un Workspace aislado en el servidor 3.16.1 completó
la carga real con el SDK 3.16.0 y el smoke `refresh → health → predict` sobre
un modelo sintético propio. En esta topología, el servidor anuncia descargas
multipart mediante URLs firmadas que apuntan al nombre interno `minio:9000`,
inaccesible para el proceso host. Se fijó
`MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false` **sólo para los procesos de
esta prueba**, conservando la carga nativa `models:/` a través del proxy de
artifacts de MLflow. No se cambió el pin, el alias productivo ni la
configuración global. El resultado no acredita la ruta multipart predeterminada
ni un despliegue diferente; cada topología debe verificar su acceso al almacén
o usar explícitamente el proxy de artifacts antes de afirmar compatibilidad.

Si el Model API se ejecuta en el host con esta topología, configura
`MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false` en el entorno del **proceso
servidor Model API antes de iniciarlo**, para que lo hereden tanto la carga
inicial como el proceso de refresh. Configurarlo únicamente en el cliente
`scripts/model_release_smoke.py` no cambia el transporte usado por el servidor.

```bash
export MLFLOW_TRACKING_URI="https://mlflow.example.internal"
export MLFLOW_TRACKING_USERNAME="..."
export MLFLOW_TRACKING_PASSWORD="..."
```

`MODEL_API_ADMIN_TOKEN` es obligatoria y protege exclusivamente el refresh
administrativo. Debe proveerse sólo mediante el entorno de ejecución; si falta,
es vacía o contiene sólo espacios, el proceso falla cerrado al iniciar. No la
incluyas en el repositorio, imágenes, logs o respuestas HTTP.

```bash
export MODEL_API_ADMIN_TOKEN="..."
```

`MODEL_REFRESH_TIMEOUT_SECONDS` es opcional y limita la preparación del nuevo
worker, incluidos spawn, carga, deserialización y recepción de metadata; su valor
por defecto sigue siendo **30 s**. Debe ser finito y positivo. Si falla o vence el
plazo, se termina y recolecta el candidato, se conserva el worker activo y se
responde `503`. La limpieza añade como máximo **0,3 s por worker** de esperas
programadas (terminate/join 0,1 s, kill/join 0,1 s y cierre IPC 0,1 s). No confirma
cancelación del proveedor ni garantiza planificación puntual del sistema operativo.

```bash
export MODEL_REFRESH_TIMEOUT_SECONDS="30"
```

`MODEL_STARTUP_TIMEOUT_SECONDS` es opcional, finito y positivo, con default
**180 s por entrada**. El arranque usa el mismo supervisor, sin ejecutar el loader
en el proceso HTTP. Los imports anteriores al lifespan quedan fuera de este plazo;
un supervisor externo debe cubrirlos. Una carga fallida degrada sólo esa entrada.

### Worker propio, inferencia y cierre

El estimador y toda deserialización permanecen en un proceso propio creado con
`spawn` durante toda la vida del modelo. El proceso HTTP retiene únicamente un
proxy con `classes_`, `predict_proba`, versión y run. Se mantiene la llamada
posicional de ocho features y la conversión numpy dentro del worker; no se
transfiere ni deserializa un pickle del modelo en el API. La carga concreta sigue
usando `models:/<modelo-registrado>/<versión>` o el proxy local existente, sin
cambiar su lista de confianza skops. Predict no consulta MLflow.

#### Privacidad de stdout/stderr del worker (CR-012)

El launcher conecta los FD 1 y 2 del hijo a `DEVNULL` mediante `subprocess.Popen`
antes de ejecutar su intérprete: cubre bootstrap, imports durante unpickle,
carga, inferencia y salida normal del intérprete. No redirige los FD del proceso
HTTP, ni siquiera temporalmente; sus eventos estructurados seguros siguen
visibles. No hay pipes de captura, buffers de retención ni lectores de salida
que puedan bloquear por volumen. Si falla la apertura del destino nulo o la
creación del proceso, falla cerrado, sin heredar stdout/stderr como alternativa.

`worker_launch.py` (80 líneas) incorpora `QuietSpawnContext`, conserva el protocolo
interno de preparación/reducción de `multiprocessing` de **CPython 3.12 en
Linux/macOS** y usa `Popen` para crear y
recolectar al hijo. No cambia el start method global ni usa fork para ejecutar
el loader en el intérprete HTTP. El payload privado viaja por pipe anónimo;
argv sólo contiene el comando de bootstrap y números de FD. Debe revalidarse
este adaptador antes de ampliar el soporte de Python o plataforma. Los pipes
JSON, reservas previas, límites de capacidad, leases y deadlines se mantienen.
El owner registra al hijo antes de escribir el bootstrap; ante fallos, los FD
nuevos se cierran o quedan bajo seguimiento de recolección. No se abre un canal
de secretos nuevo por argv ni entorno. En `app.py` sólo cambian el import y la
factoría del contexto; `worker.py` y sus pruebas anteriores quedan intactos.
CR012 no cambia dependencias, pins, HTTP público, autenticación, confianza ni
fuentes de Stack/MLflow.

Las regresiones de `tests/test_worker_output.py` capturan stdout/stderr desde un
subproceso exterior con workers reales: imports anteriores al target, `print`,
logging, `os.write`, streams originales, errores, `SystemExit`, salida `atexit`
confirmada y volumen superior a la capacidad habitual de un pipe. Comprueban
marcadores concurrentes completos del proceso HTTP, startup/refresh/predict HTTP,
metadata y probabilidad, cierre concurrente, cancelación y fallos de recursos,
incluida retención de un candidato cuya recolección no pudo confirmarse.

Este límite cubre **stdout/stderr heredados**, no constituye un sandbox ni una
garantía de borrado universal: un proveedor confiable puede abrir sus propios
archivos o canales de red. No garantiza zeroization ni seguridad frente a un
modelo malicioso. El protocolo privado está acotado a CPython 3.12 POSIX
Linux/macOS: no garantiza Windows ni Python 3.13; revalidar antes de ampliarlo.

**Cierre CR012 (2026-10-19): RESOLVED_LOCAL; revisión independiente
READY_WITH_FOLLOW_UP**, sin `MUST_FIX` restante, pero con CR013 abierto. Evidencia
recibida, no reejecutada en esta conciliación; cada corrida se informa por separado:

| Gate | Resultado observado |
| --- | --- |
| HOST macOS, CPython 3.12.8, RED previo | 6 fallos causales de canarios, **6.95 s**, antes del fix. |
| HOST, regresiones nuevas | **12 PASS, 6.01 s**; `test_worker_output.py`, 327 líneas, captura real exterior de FD. Incluye `sys.__stdout__`/`sys.__stderr__`, `atexit`, controles concurrentes, fallos de destino/exec/pipe, estabilidad FD, retención de owner y `TestClient` con metadata/probabilidad **0.8**. |
| HOST, suite completa | **168 PASS, 55.46 s**: 156 anteriores + 12 nuevas; salida 0. |
| HOST, estática | Ruff, formato de los **11 archivos Python** del cambio acumulado, AST y whitespace **PASS**; no implica formato global aprobado (deuda previa API-06 CR013, distinta del CR013 de STACK-02). |
| HOST, revisión independiente | **12 PASS, 6.93 s**; señales/salidas **0/7/-15/-9**, comprobación **0.202 s**. |
| HOST, comprobación final del responsable | **12 PASS, 7.82 s**. |
| Build de producción Linux | `invoiceops-cr012-check:20261019`, salida **0**, **406.431 s**; imagen `sha256:ca9a461c58a48847d30919d56080d7f269c3217ae88436686c40ff8fc3930e32`. Linux **6.10.14 aarch64**, glibc **2.41**, CPython **3.12.15**, `USER app`, UID/GID **999/999**. |
| Primera sonda Linux | Salida **1**, **14.146 s**: aserción privada esperaba `prediction_failure` en vez del contrato `inference_failure`. No acredita defecto productivo ni se oculta el fallo. |
| Sonda Linux corregida, misma imagen | **6/6 PASS**, salida **0**, **30.623 s**: `success`, `load_error`, `load_exit`, `predict_error`, `predict_exit`, `import_error`. Sólo se corrigió el driver privado; sin rebuild ni cambios productivos. |

En Linux, `worker_launch.py` coincide con HOST: SHA-256
`7cfc5f73bb5b98b193958b51b1f8bcd986e75ee064bc8afed16f45df664f9edb`.
Proxy real: `classes_=[0,1]`, ocho campos, `predict_proba=[[0.2,0.8]]`, versión
`7`, run `run-123`. En los seis casos no aparecen canarios en ninguno de los dos
streams y permanecen visibles los controles del proceso responsable; pool vacío
y PIDs propios ausentes al cerrar. Liveness confirmada en los cinco casos que
alcanzan el loader (`import_error`: no aplica); volumen alto y salidas
**0/7/-15/-9** comprobados. No se atribuye `atexit` a Linux ni una suite pytest
Linux de 12 casos: `atexit` está confirmado sólo por HOST.

La sonda usó `stack02-cr012-linux-20261019`, `--rm`, red `none`, raíz de sólo
lectura y tmpfs **128 MiB**, sin puertos, volúmenes ni binds. Ausencia posterior
del nombre exacto confirmada; imagen preservada. Producción no incluye pytest
(`--no-dev`); no se instalaron dependencias nuevas ni se tocaron servicios,
reinicios o recursos ajenos. **NOT_VERIFIED:** wheel, CI remoto, nuevo full-stack
u ops08 con este launcher, soporte universal de versiones y SLO. El ops08
histórico de **11 PASS / 556.121 s** no prueba estos bytes.

| Operación | Límite y comportamiento |
| --- | --- |
| IPC | Dos pipes anónimos no bloqueantes; header de 4 bytes y JSON de hasta 16 KiB, sólo escalares/listas, profundidad y tamaños acotados. Metadata, clases y resultados se validan antes de publicarse. Tramas parciales, corruptas, excesivas o tardías fallan sin `Connection.recv()` ni unpickle. |
| Inferencia | **10 s internos**, incluida espera del mutex, escritura y lectura. Un RPC por worker evita cruzar respuestas. Un RPC vencido en curso, transporte corrupto o salida del hijo invalida y recolecta ese worker; predict y salud del modelo pasan a `503`. Recuperar mediante refresh autorizado. Un error completado del estimador devuelve `503` sin excepción cruda y permite reutilizar el worker vivo. |
| Refresh concurrente | Sólo uno; otro intento recibe `503` inmediato en lugar de esperar sin límite. Carga fallida conserva modelo/run activos. |
| Snapshot anterior | Predict adquiere una lease bajo el lock del registro, antes del RPC. Tras publicar el nuevo worker, la solicitud anterior conserva su proxy y metadata. Se retira al liberar la última lease o, como límite de drenaje, a los **10 s desde el swap**, más limpieza. Una solicitud que exceda ese drenaje falla de forma segura. |
| Capacidad | Máximo **3 × entradas del catálogo** de owners vivos, contando activo, retirados y candidato. Si no hay capacidad, refresh responde `503` sin crear otro hijo. No depende de GC para retirar modelos. |
| Shutdown | Bloquea nuevas publicaciones, cancela el candidato y espera hasta **0,5 s** su supervisor; cierra activos y retirados incluso con leases, sin esperar el mutex RPC para terminar al hijo. Presupuesto de esperas: **0,5 + 0,3 × owners s** (1,4 s con el catálogo actual y tres owners). Cancelar startup repetidamente no separa al supervisor de su limpieza. |

El IPC admite strings de hasta 2048 caracteres, listas de hasta 64 elementos y
números finitos de magnitud máxima `1e100`, además del presupuesto total previo
a codificar. Inputs válidos para el schema HTTP pero fuera del límite de transporte
no se envían al estimador y reciben `503` sin invalidar el worker; no se añade una nueva regla `422`.
La API sigue siendo un servicio de una fila, no una API de batch arbitrario.

La limpieza comprueba terminación antes de cerrar el handle. Si el SO no permite
confirmar terminación/recolección, se informa `cleanup_unverified`, nunca una
promesa de ausencia de procesos. Un owner retirado no verificado permanece en la
lista acotada, registra `model_cleanup_failed` y se reintenta al cerrar; una
publicación ya confirmada no se revierte por ese fallo posterior de limpieza.
Estos procesos aíslan bloqueos, **no son un sandbox para artifacts no confiables**
ni un límite de RSS del estimador. El pico puede incluir hasta tres estimadores
e intérpretes; dimensionar memoria con el artifact real antes de desplegar.

Las pruebas offline de `tests/test_model_worker.py` usan hijos reales y pipes,
incluido startup/refresh/predict con `TestClient`, tramas incompletas, deserialización
e inferencia bloqueadas, concurrencia, leases y cierre. El fixture HGB posicional
de 64 filas compara outputs con el estimador directo y publica `OFFLINE_WORKER_COST`
con latencia y RSS local mediante `ps`. RSS incluye páginas compartidas, puede variar
por presión de memoria y no equivale a PSS ni a un pico de producción. Esta evidencia
local no sustituye el smoke operacional: CR007 está ahora cerrado por ops08
estándar fresco PASS, no por estas mediciones offline.

## Smoke de model release

El smoke es una verificación explícitamente opt-in para el operador después de
una promoción manual. Ejecuta `refresh → health → predict` para
`invoice-review`; no inicia el servicio, no cambia aliases ni contacta MLflow
directamente.

Provee exclusivamente por entorno la URL de la instancia, el token
administrativo y la metadata esperada:

```bash
export MODEL_API_BASE_URL="https://model-api.example.internal"
export MODEL_API_ADMIN_TOKEN="..."
export MODEL_RELEASE_EXPECTED_MODEL_VERSION="..."
export MODEL_RELEASE_EXPECTED_RUN_ID="..."
uv run python scripts/model_release_smoke.py
```

Cada solicitud HTTP se ejecuta en un hijo aislado con plazo operacional total
para conexión, transferencia y lectura: **10 s por defecto**. La variable opcional
`MODEL_RELEASE_HTTP_TIMEOUT_SECONDS` admite un valor finito positivo de hasta
**125 s**; STACK-02 usa explícitamente **125 s** frente al refresh local de 120 s,
sin alterar el default global. El arnés Stack añade un límite exterior de
**400 s** para las tres solicitudes. Un servidor que envíe bytes periódicos no
puede prolongar indefinidamente el plazo. Al vencerlo, el padre termina al hijo,
espera como máximo un segundo y, sólo si sigue vivo, lo mata y espera otro
segundo antes del fallo seguro. El hijo no escribe stdout/stderr; el temporal
contiene sólo la respuesta HTTP y se elimina antes de continuar. El comando no
imprime URL ni token. Configuración incompleta, HTTP fallido, metadata inesperada,
respuesta inválida o probabilidad fuera de rango producen salida no cero.

## Verificación local

```bash
uv run pytest
uv run ruff check .
uv build
```

## CI del repositorio (API-07, fase local)

`.github/workflows/ci.yml` declara una verificación en pull requests hacia
`main`, pushes a `main` y ejecución manual. Usa Python 3.12 y uv 0.12.5;
comprueba el lockfile, instala dependencias y herramientas de desarrollo con
`uv sync --locked --all-groups`, ejecuta **toda** la suite pytest y
`ruff check .`, y construye sdist y wheel con el backend Hatchling declarado.
Cada paso falla normalmente si su comando falla; el job tiene un límite de
20 minutos. La instalación de herramientas y dependencias ocurre sólo cuando
un runner ejecute el workflow, no por agregar este archivo.

El workflow tiene permiso `contents: read`, no conserva credenciales de checkout,
no configura secretos ni credenciales Cloud Run/MLflow y no publica imágenes,
despliega servicios, entrena modelos ni modifica el alias `champion`. El formato
global **no** es un gate de CI: CR013 identifica cuatro archivos intactos que
no superan `ruff format --check .`. Para Python modificado, verificar formato
sólo en los archivos del cambio hasta resolver ese seguimiento por separado;
no presentar el check global como aprobado. Esta fase sólo declara CI: todavía
no hay ejecución del workflow en GitHub, publicación de digest ni validación
de configuración, salud de proceso/modelo o rollback en Cloud Run.

## Container and local Compose ownership

The production image is built from this repository's `Dockerfile` and `uv.lock`.
The runtime stage copies only the production virtual environment and `src`, runs
as `app` (not root) on port 8000, and sets bounded BLAS thread counts. The
`.dockerignore` excludes local environment files and development artifacts.
These are source-level properties; a successful build does not establish
running-image functionality.

After authorization and verification that the local Docker context is the
intended one, build from this repository with `docker build -t
invoiceops-model-api .`. This may pull base images, install production packages,
consume disk space and write to the local image store. The disposable local
attempt below used a unique tag instead of overwriting this example tag.

**Evidencia operativa parcial (2026-10-07):** con autorización para una
instalación local nueva y aislada, se verificó el socket Docker local
`unix:///var/run/docker.sock`. El build de Model API terminó correctamente en
282,25 segundos; la inspección de configuración de la imagen confirmó
`USER app` y el puerto `8000/tcp`. No se comprobó el UID efectivo, la
importación de dependencias ni el funcionamiento de esa imagen en ejecución.
MLflow y la base App nuevos alcanzaron estado saludable; migración, carga de
propietario sintético y entrenamiento RF terminaron con código 0. El helper
se detuvo al interpretar como JSON toda la salida del entrenamiento
(`JSONDecodeError`); no se conservó esa salida y su causa exacta es desconocida.
No se ejecutaron el Gate, los notebooks, la promoción, el inicio de Model API,
la salud del modelo ni el smoke `refresh → health → predict`.

Se retiraron los contenedores, volúmenes y redes de los proyectos propios, la
red externa propia, la etiqueta de imagen única y los archivos temporales
privados; el inventario final no encontró recursos nuevos. Los conteos de
redes y volúmenes preexistentes no cambiaron, pero el de contenedores pasó de
25 a 24: no se conservaron los identificadores iniciales y no se puede
identificar ni atribuir la desaparición. La consulta local de eventos de
contenedores entre 17:30 y 17:43 UTC no devolvió registros; no demuestra su
causa. Recibo sanitario: `api06-receipt-1b479ee0d7.json` en el directorio
privado temporal de evidencias, fuera de los repositorios. Antes de otra
operación se requiere una decisión sobre esta diferencia y sobre la captura
segura de identificadores/etiquetas y metadatos de parsing, sin guardar la
salida cruda ni credenciales. Una nueva ejecución no queda autorizada por
este registro.

**Segundo intento local, también parcial (2026-10-07):** una instalación
aislada nueva volvió a construir la imagen (81,70 segundos; ID
`sha256:df50c128c325c97f2cee5f934818911dc1108ae7965284e0bffc661e144ae615`,
`USER app`, `8000/tcp`). Se preservaron inventarios privados con ID, nombre,
estado, imagen y etiquetas de propiedad antes y después. MLflow y la base App
alcanzaron salud; migración y carga sintética terminaron correctamente. La
función real `train_candidate` generó un reporte estructurado para el run
`bfcad5dcd1c440b2873f045048d2c082` (44,91 segundos), sin depender del
stdout mezclado del CLI. Su Gate se ejecutó y devolvió **FAIL** (código 1):
RF no es elegible para la promoción local. El notebook 03 terminó con código 0;
el notebook 04 terminó con código 1 después de 309,91 segundos. No se
conservó el error privado y no se establece si fue timeout u otra falla. No
se ejecutaron notebook 05, Gate HGB, promoción, inicio de Model API, inspección
de UID efectivo ni smoke. El transporte `proxy-http` estaba configurado para
la topología, pero **no** se verificó en el proceso Model API.

Se eliminaron los contenedores y volúmenes propios; una comprobación posterior
confirmó también la ausencia de la red externa propia y de ambas etiquetas de
imagen exclusivas, retiradas por ID/nombre exactos tras detectar un defecto
del helper de limpieza. Los inventarios completos inicial/final y el recibo
sanitario `api06-proof-8c23b8ef36.json` permanecen en el directorio privado
temporal de evidencias; `api06-8c23b8ef36-postcleanup.json` documenta la
comprobación posterior sólo del namespace propio. Durante este intento,
tres contenedores ajenos del proyecto `stablepro-chat` desaparecieron y se
observaron tres reemplazos con los mismos nombres de servicio; no hay evidencia
del actor o de su causa y no fueron intervenidos. La desaparición del
contenedor del intento anterior **sigue sin resolverse**. El fallo del
notebook 04 es histórico y no bloquea una prueba aislada del loader de API-06.

**Corrección de alcance y prueba local acotada (2026-10-07):** los intentos
anteriores de entrenamiento/Gate/notebooks no eran necesarios para probar el
loader de API-06 y no deben repetirse para este objetivo. Se registró
exclusivamente en un MLflow nuevo y descartable un modelo sintético sklearn
compatible con el loader (`invoice-review-production@champion`, versión `1`),
sin Gate ni promoción de release. El cliente local MLflow 3.16.0 confirmó
`READY` para el run `0f369fbf72f04037900f8581100a188a`. Se reutilizó una
imagen de producción **preexistente** (`sha256:c158ad4a26b2c46cb1fcd012b2ea825a830185f33bd8c772ff6f9acdb865ae31`)
tras comparar el SHA-256 de `app.py` dentro de ella con el checkout. Ejecutó
como UID **999** con `proxy-http` y cargó el modelo: `GET
/models/invoice-review/health` respondió con versión `1` y el run esperado.
El smoke real `scripts/model_release_smoke.py` **falló** en `POST
/admin/models/invoice-review/refresh` con HTTP **503**; por ello no se
comprobaron `health` posterior al refresh ni `predict` en este intento.
La causa interna del 503 no quedó capturada; no se atribuye al modelo, al
transporte ni a credenciales sin evidencia. Un segundo build nuevo agotó su
límite de 360 s, aunque los dos builds de intentos anteriores habían aprobado.
La imagen reutilizada prueba carga en ejecución, pero no constituye una
reconstrucción exitosa en este intento. Recibo sanitizado:
`/var/folders/y3/m1xh54ds39x7c15dzzngkh1r0000gn/T/opencode/api06-direct-6c7550084f-receipt.json`.
Sus snapshots inicial/final indican ausencia de recursos propios y los mismos
IDs de contenedores, redes y volúmenes preexistentes. La diferencia histórica
de un contenedor del primer intento continúa sin explicación. El objetivo
local permanece **parcial**: un modelo sintético prueba carga, NO elegibilidad
de release, Gate, ruta remota ni smoke completo. Este párrafo conserva el
resultado histórico; la prueba más reciente se registra a continuación.

**Resultado local vigente (2026-10-07):** se reutilizó la misma imagen
preexistente (`sha256:c158ad4a26b2c46cb1fcd012b2ea825a830185f33bd8c772ff6f9acdb865ae31`),
con `app.py` idéntico al checkout y UID efectivo `999` (MLflow 3.16.0,
sklearn 1.9.1, skops 0.16.0 en la imagen). En un namespace nuevo, el modelo
sintético de MLflow cargó al arrancar y `predict` devolvió **0,5** con el run
y la versión realmente activos. El primer `refresh` devolvió 503 a los
34,016 s; la carga aislada reprodujo `ModelLoadFailure(timeout)` a los
30,023 s. El proceso hijo llegó a `client.list_artifacts` en menos de 3 s y
quedó bloqueado. El MinIO **propio** terminó dos veces por OOM (exit 137);
MLflow respondió HTTP 500 para artifacts, con `EndpointConnectionError` y
`NameResolutionError` de `minio`. Reiniciar MinIO no resolvió de forma estable
el problema. No se modificaron el timeout, la autenticación, el loader ni el
transporte de producción.

Para aislar la prueba de conectividad API ↔ MLflow del fallo de memoria de
MinIO, **sólo en el proyecto descartable**, se recreó MLflow con destino de
artifacts de sistema de archivos efímero y se registró un nuevo artifact
skops real: `invoice-review-production@champion`, versión `2`, run
`5f18ebc598f243b9b879e71b9bdcc88f` (`READY`). En la **misma imagen y
proceso** Model API, el refresh autenticado respondió `200` en 6,175 s,
`health` y `predict` posteriores informaron esa versión/run y probabilidad
`0,5`; `scripts/model_release_smoke.py` aprobó `refresh → health → predict`
(exit 0). El refresh sin token respondió `401` sin cambiar el modelo activo.
Esto acredita el loader, la inferencia y el transporte local `proxy-http`
contra MLflow, **no** la ruta S3/MinIO del overlay normal ni Gate/release
canónica o conectividad remota. El destino efímero fue exclusivamente un
override de prueba fuera del repositorio; la configuración versionada no
cambió. Se retiraron el contenedor API, cuatro contenedores Compose, dos
volúmenes y la red externa propios; los conjuntos de ID de contenedores,
redes y volúmenes preexistentes permanecieron iguales (60/18/177). Recibo
sanitizado: `/var/folders/y3/m1xh54ds39x7c15dzzngkh1r0000gn/T/opencode/api06-final-b6209f3a18-receipt.json`.
La desaparición histórica de un contenedor ajeno sigue sin explicación.

Do not create a second Compose stack here. The existing local Model API service,
build context, loopback port, model-loaded healthcheck and credentials wiring are
owned by [`invoiceops-app/compose.invoiceops-local.yml`](../invoiceops-app/compose.invoiceops-local.yml).
The MLflow overlay is owned by
[`invoiceops-mlflow/compose.invoiceops-local.yml`](../invoiceops-mlflow/compose.invoiceops-local.yml).
For an **authorized, fresh, disposable local installation**, follow the
[`LOCAL_INVOICEOPS_RUNBOOK.md`](../invoiceops-mlflow/LOCAL_INVOICEOPS_RUNBOOK.md)
from start to finish **for a full release demonstration**. Its prerequisite
checks, local-only network, isolated project names, synthetic candidate Gate,
explicit promotion and expected version/run capture apply to that release
workflow, not to a disposable loader-only fixture.
Never point that procedure at an existing or shared deployment. Infra owns the
MLflow installation; App owns the Compose overlay; Model API owns its image,
configuration, endpoints and release smoke script.

### Receta mínima aislada: sólo carga del modelo

Esta receta **no se ejecutó como parte de esta documentación**. No es una
demostración de release: omite App, PostgreSQL App, propietarios académicos,
notebooks, entrenamiento candidato, Quality Gate y promoción operativa. Requiere
autorización para crear recursos locales nuevos y descartables; no reutilices un
MLflow, alias, red, volumen, puerto o credencial existente. Reserva los puertos
libres `5500` y `5800` en loopback; elige nombres exclusivos, inventaría su
ausencia antes de crearlos y exporta `LOCAL_NETWORK`, `LOCAL_MLFLOW_CONTAINER`,
`LOCAL_API_CONTAINER`, `VERIFIED_MLFLOW_IMAGE` y `VERIFIED_MODEL_API_IMAGE`.
Ambas imágenes deben estar **ya construidas**; coteja la de Model API con este
checkout antes de usarla. Esta receta no construye imágenes ni crea otro
Compose persistente. Prepara en un entorno privado nuevo
`MLFLOW_AUTH_ADMIN_USERNAME`, `MLFLOW_AUTH_ADMIN_PASSWORD`,
`MLFLOW_FLASK_SERVER_SECRET_KEY` y `MODEL_API_ADMIN_TOKEN`; no uses valores
en argv ni un `.env` anterior.

1. Desde `invoiceops-model-api`, crea la red exclusiva e inicia el contenedor
   MLflow propio en una terminal. La imagen MLflow debe ser la existente de
   `invoiceops-mlflow/Dockerfile.mlflow`, cuyo entrypoint admite los stores
   SQLite nuevos; metadata, auth y artifacts viven **sólo** en el sistema de
   archivos efímero del contenedor. Comprueba el `401` anónimo y la respuesta
   autenticada antes de registrar el fixture. Si el contenedor no queda sano,
   detente; no sustituyas esto por el overlay habitual S3/MinIO.

   ```bash
   docker network create "$LOCAL_NETWORK"
   docker run --rm --name "$LOCAL_MLFLOW_CONTAINER" \
     --network "$LOCAL_NETWORK" --network-alias mlflow \
     -p 127.0.0.1:5500:5000 \
     --env MLFLOW_AUTH_ADMIN_USERNAME --env MLFLOW_AUTH_ADMIN_PASSWORD \
     --env MLFLOW_FLASK_SERVER_SECRET_KEY \
     --env MLFLOW_AUTH_CONFIG_PATH=/tmp/mlflow-loader/auth.ini \
     --env MLFLOW_AUTH_DATABASE_URI=sqlite:////tmp/mlflow-loader/auth.db \
     --env MLFLOW_BACKEND_STORE_URI=sqlite:////tmp/mlflow-loader/tracking.db \
     --env MLFLOW_SERVER_ALLOWED_HOSTS=mlflow:5000,127.0.0.1:5500 \
     "$VERIFIED_MLFLOW_IMAGE" mlflow server --app-name=basic-auth \
     --host 0.0.0.0 --port 5000 --serve-artifacts \
     --artifacts-destination /tmp/mlflow-loader/artifacts
   ```

2. En otra terminal, desde `invoiceops-model-api`, exporta
   `MLFLOW_TRACKING_URI=http://127.0.0.1:5500`,
   `MLFLOW_TRACKING_USERNAME` y `MLFLOW_TRACKING_PASSWORD` con los valores de
   admin de **esta** instancia, y deja `MLFLOW_WORKSPACE` ausente. El cliente
   Python 3.12 usa `mlflow==3.16.0`, scikit-learn y skops. Crea
   **sólo** el fixture en esa instancia (el nombre canónico está permitido
   únicamente porque la instancia es nueva y exclusiva):

   ```bash
   uv run python - <<'PY'
   import mlflow
   import mlflow.sklearn
   import pandas as pd
   from tempfile import TemporaryDirectory
   from mlflow import MlflowClient
   from sklearn.compose import ColumnTransformer
   from sklearn.ensemble import HistGradientBoostingClassifier
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

   rows = pd.DataFrame({
       "invoice_amount_cents": [125000 + i * 1000 for i in range(32)],
       "vendor_tenure_days": [365 + i for i in range(32)],
       "previous_incidents_12m": [i % 2 for i in range(32)],
       "amount_vs_vendor_median": [1.1] * 32,
       "has_purchase_order": [True] * 32,
       "three_way_match": [True] * 32,
       "bank_account_recently_changed": [False] * 32,
       "country_risk": [("low", "medium", "high")[i % 3] for i in range(32)],
   })
   model = Pipeline([
       ("encode", ColumnTransformer(
           [
               ("numeric", "passthrough", [0, 1, 2, 3]),
               ("flags", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [4, 5, 6]),
               ("risk", OrdinalEncoder(), [7]),
           ]
       )),
       ("classifier", HistGradientBoostingClassifier(
           min_samples_leaf=2, random_state=0
       )),
   ])
   model.fit(rows, [i % 2 for i in range(32)])
   with mlflow.start_run() as run:
       with TemporaryDirectory() as model_dir:
           mlflow.sklearn.save_model(model, model_dir, serialization_format="skops")
           mlflow.log_artifacts(model_dir, artifact_path="model")
   client = MlflowClient()
   version_info = mlflow.register_model(
       f"runs:/{run.info.run_id}/model", "invoice-review-production"
   )
   version = version_info.version
   assert version_info.run_id == run.info.run_id
   assert client.get_model_version("invoice-review-production", version).status == "READY"
   client.set_registered_model_alias("invoice-review-production", "champion", version)
   print("MODEL_RELEASE_EXPECTED_MODEL_VERSION=", version)
   print("MODEL_RELEASE_EXPECTED_RUN_ID=", run.info.run_id)
   PY
   ```

   Se ajusta con columnas tipadas del `DataFrame`, pero el Model API llama a
   `predict_proba` con una lista posicional; por eso el selector de
   `country_risk` debe ser `[7]`, **no** `['country_risk']`. El fixture sólo
   demuestra carga y smoke, no fidelidad de las features categóricas de negocio.
   El modelo sintético no tiene Gate aprobado. Comprueba en la instancia propia
   que el artifact del run contiene `model/MLmodel` y `model/model.skops`; el
   loader local restringe los tipos skops admitidos. Si falla, detente sin
   cambiar la lista de confianza de producción ni un alias ajeno.

3. Desde `invoiceops-model-api`, con la red e imagen exclusivas ya verificadas,
   inicia **sólo** el proceso Model API; inyecta las variables desde el entorno
   privado. Exporta `MLFLOW_TRACKING_URI=http://mlflow:5000` para **este**
   contenedor (la URI loopback del cliente host no sirve en Docker). Sustituye
   los marcadores de nombres por identificadores propios comprobados, no por
   los de un entorno compartido:

   ```bash
   docker run --rm --name "$LOCAL_API_CONTAINER" --network "$LOCAL_NETWORK" \
     -p 127.0.0.1:5800:8000 \
     --env MLFLOW_TRACKING_URI --env MLFLOW_TRACKING_USERNAME \
     --env MLFLOW_TRACKING_PASSWORD --env MLFLOW_WORKSPACE \
     --env MODEL_API_ADMIN_TOKEN \
     --env MODEL_API_ARTIFACT_DOWNLOAD_MODE=proxy-http \
     "$VERIFIED_MODEL_API_IMAGE"
   ```

   Las credenciales Basic Auth deben pertenecer sólo al MLflow desechable;
   `MLFLOW_WORKSPACE` debe permanecer ausente para este fixture del Workspace
   predeterminado. `GET /health` no basta:
   comprueba `GET http://127.0.0.1:5800/models/invoice-review/health` y exige
   `200` con el run y versión obtenidos en el paso 2. Si devuelve `503`, no
   ejecutes el smoke.

4. En otra terminal de `invoiceops-model-api`, con el token privado y los
   **valores observados** del paso 2 exportados en el entorno, verifica el mismo
   proceso y puerto mediante el script existente:

   ```bash
   export MODEL_API_BASE_URL="http://127.0.0.1:5800"
   uv run python scripts/model_release_smoke.py
   ```

   Se requieren `MODEL_API_ADMIN_TOKEN`,
   `MODEL_RELEASE_EXPECTED_MODEL_VERSION` y `MODEL_RELEASE_EXPECTED_RUN_ID`.
   El script realiza `refresh → health → predict` y falla si no coincide la
   metadata o la probabilidad no es válida. Detén únicamente los dos
   contenedores **creados para esta prueba** por sus nombres exactos
   previamente inventariados y elimina después sólo su red exclusiva. No hay
   volúmenes persistentes en este ejemplo; `--rm` retira los archivos efímeros
    de los contenedores al detenerlos. No uses `docker volume prune`, `down -v`
    sobre un proyecto compartido ni credenciales de otra instalación. La ruta S3/MinIO habitual,
   la descarga SDK remota y la identidad de un build nuevo **no quedan probadas**
   por esta receta ni por el smoke local anterior.

The runbook's Model API step builds the image and starts the dedicated App
project with `up --build --wait --wait-timeout 180 model-api backend frontend`.
Its healthcheck requests `GET /models/invoice-review/health`; an unhealthy
model must STOP the workflow before smoke. `GET /health` reports only process
availability and does not prove that an artifact loaded. Once the local
operator has verified the target and obtained `MODEL_VERSION` and `RUN_ID`
from the authorized synthetic promotion, the runbook exports the loopback
`MODEL_API_BASE_URL`, supplies `MODEL_API_ADMIN_TOKEN` from the private local
environment, exports `MODEL_RELEASE_EXPECTED_MODEL_VERSION` and
`MODEL_RELEASE_EXPECTED_RUN_ID`, then runs this command from the runbook's
`invoiceops-mlflow` working directory:

```bash
( cd ../invoiceops-model-api && MODEL_RELEASE_HTTP_TIMEOUT_SECONDS=45 python3 scripts/model_release_smoke.py )
```

This opt-in script performs authenticated `refresh → model health → predict` and
requires the exact expected version/run and a finite probability. It changes
the active model in the selected Model API process; do not run it against a
shared target. The local overlay opts into `proxy-http` for this isolated
network only; see the runbook for its transport limits. A previous host-local
SDK test using `MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false` does NOT prove
that the overlay's `proxy-http` mode or the SDK default succeeds in a container.

For a separately authorized remote tracking service, use the **same image**,
with `MODEL_API_ARTIFACT_DOWNLOAD_MODE` omitted (SDK default). An operator must
first approve the destination, Model API instance, model/alias and credential
channel, and verify TLS trust, DNS/routing, MLflow permissions and artifact
download reachability from **inside the container**. On a trusted operator
shell, inject credentials from a private runtime environment, not command-line
values or a build argument. For example, once those variables have been
supplied privately and checked, this command passes only their names:

```bash
docker run --rm --name invoiceops-model-api-check -p 127.0.0.1:8000:8000 \
  --env MLFLOW_TRACKING_URI --env MLFLOW_TRACKING_USERNAME \
  --env MLFLOW_TRACKING_PASSWORD --env MLFLOW_WORKSPACE \
  --env MODEL_API_ADMIN_TOKEN \
  invoiceops-model-api
```

Basic Auth variables must both be present or both absent; the tracking URI
and admin token are mandatory; `MLFLOW_WORKSPACE` is optional and must refer
only to an authorized Workspace. Use an approved host port and container name
only after checking they are free, and restrict access to loopback. The
operator must then check `/health`, `/models/invoice-review/health` and the
release smoke with expected version/run in the **same approved remote target**.
Do not infer remote readiness from a healthy process or from local-only proxy
settings; signed artifact URLs may require a different network path. The
remote example is a procedure, NOT a recorded runtime PASS.

On a failed load, stop before invoking the release smoke; inspect only
sanitized service status and logs, correct the connectivity or authorization
problem, then retry in the approved boundary. For rollback, stop only the
Model API container started by this procedure (`docker stop
invoiceops-model-api-check` if it is still running), or restore the prior
Model API image/configuration in the dedicated project after explicit
operational approval. Do not remove MLflow, DB/MinIO, shared networks or
volumes; do not run `down -v`, promote another alias, or alter auth policy.
La carga, el refresh y el predict locales aprobaron con el fixture sintético
y artifacts efímeros; la ruta S3/MinIO normal falló por OOM local y el smoke
remoto de API-06 sigue sin verificarse.
AC03 remoto requiere autorización independiente del destino, identidad/canal,
modelo/alias y operaciones permitidas; este intento sólo usó el Docker local.

## Alcance actual

- Incluye `GET /health`.
- Incluye `GET /models`, que lista los IDs lógicos registrados en orden.
- Incluye las rutas genéricas `GET /models/{model_id}/health` y `POST /models/{model_id}/predict`; agregar un ID al catálogo no requiere agregar rutas FastAPI.
- El catálogo local tipado mapea cada ID lógico a un modelo registrado y a un alias no vacío:

| ID lógico        | Modelo registrado           | Alias      |
| ---------------- | --------------------------- | ---------- |
| `invoice-review` | `invoice-review-production` | `champion` |

- Un ID desconocido responde `404`. Un ID registrado que no pudo cargar responde `503`; el detalle de su respuesta de health incluye el ID lógico, el modelo registrado, el alias y un mensaje de loader no disponible. Fallas del proveedor, alias, metadata o carga degradan sólo el ID afectado y no exponen el error del proveedor.
- `POST /models/{model_id}/predict` exige exactamente las ocho features del artifact entrenado: `invoice_amount_cents`, `vendor_tenure_days`, `previous_incidents_12m`, `amount_vs_vendor_median`, `has_purchase_order`, `three_way_match`, `bank_account_recently_changed` y `country_risk`. Un body inválido responde el `422` estándar de FastAPI.
- Una predicción exitosa usa únicamente el modelo activo ya cargado en memoria y devuelve `model_id`, `model_version`, `run_id` y `probability`. `probability` es la probabilidad finita de la clase positiva (`1`, revisión manual); no aplica thresholds, recomendaciones ni Business Policy. La solicitud no resuelve aliases, carga artifacts ni llama a MLflow. Una salida o falla de inferencia no válida responde `503` con un detalle genérico.
- El contrato HTTP App ↔ Model API `v1`, con request/response de ejemplo, tipos, compatibilidad, responsabilidades y fallback del consumidor, se documenta en [`../invoiceops-app/README.md`](../invoiceops-app/README.md#contrato-app--model-api-int-01-v1). `country_risk` se recibe como string estricto: `low`, `medium` y `high` son los valores entrenados que envía la aplicación, pero este endpoint no restringe el dominio a esos tres valores.
- `GET /models/{model_id}/health` responde `200` cuando el modelo está activo y devuelve `model_id`, `registered_model`, `alias`, `model_version` y `run_id` realmente servidos. Un ID registrado sin carga sigue respondiendo `503`; un ID desconocido responde `404`. `GET /health` conserva su respuesta de proceso `200` con `{"status":"ok"}`.
- `POST /admin/models/{model_id}/refresh` exige el header `X-Model-Admin-Token` que coincida con `MODEL_API_ADMIN_TOKEN`. Resuelve y carga el champion actual en un proceso aislado antes de sustituir bajo lock la instancia activa. Las solicitudes predict/health sólo obtienen una instancia completa; timeout, salida anómala del hijo, error de transferencia o falla del proveedor conservan la versión anterior y responden `503` sin detalles del proveedor. Registra eventos seguros de intento, éxito o falla sin token, credenciales ni excepción cruda del proveedor.
- `scripts/model_release_smoke.py` es una verificación opt-in operada por entorno para `refresh → health → predict`; valida la versión/run esperados y una probabilidad finita entre 0 y 1, sin imprimir URL ni token.
- Local Compose is provided by the existing App and MLflow overlays linked above; only repository CI is declared for API-07. Cloud deployment remains out of scope. The application consumer is documented in INT-01, and INT-03 was absorbed by APP-08.
