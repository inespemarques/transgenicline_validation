# Scripts vs. tese: alterações necessárias

Sep 29, 2026 · @ines

A tese (draft 08/04) é a referência. Há 23 alterações a fazer nos scripts, 4 scripts em falta e 6 contradições dentro da própria tese que têm de ser resolvidas antes de mexer no código.

As mais importantes são as do classificador (`overlap_full.py`), porque decidem os números da Tabela 4.3. Nada foi alterado no repositório; comparei os scripts do branch `main` (29/09/2026) com a tese.

## Pré-processamento (`01_preprocessing/`)

O notebook atual faz outra correção em Z e numa ordem diferente da tese. Os scripts do PR #1 já seguem a tese, mas foram reconstruídos a partir do texto.

| Script | O que faz agora | O que a tese diz | Alteração | Secção |
| --- | --- | --- | --- | --- |
| `bleachcorrection.ipynb` | Correção por rácio: cada slice é multiplicado por mediana / média do slice (suavizada em 5 slices) | Histogram matching slice a slice ao slice do meio | Substituir por `zattenuation_correction.py` (PR #1) e retirar ou arquivar o notebook | 3.5, A.5 |
| `bleachcorrection.ipynb` | Lê `c2downscaledbacksubtraction.tif`: a subtração de fundo já foi feita antes da correção em Z | Downscale → correção em Z → rolling ball → mediana 3D → CLAHE → normalização | Correr pela ordem da tese (já é a ordem do PR #1) | 3.5 |
| `fiji_macros/*.ijm` (PR #1) | Escritos a partir do texto da tese, nunca corridos nos dados | Macro Fiji com rolling ball 60 px, mediana 3D r = 1, CLAHE 127 / 256 / 3, normalização linear | Correr num stack e comparar com o output publicado, ou prôr a macro original | 3.5 |
| `fiji_macros/step3_*.ijm` (PR #1) | Normalização com saturação 0,35 % (valor por defeito do Fiji) | "Linear histogram stretching to a common dynamic range", sem valor | Confirmar o valor usado e escrevê-lo na tese e no script | 3.5 |

A tese também não diz a versão do Fiji, se o Z foi reamostrado e se o CLAHE usou o modo "fast". Ver [DISCREPANCIES.md](https://github.com/inespemarques/transgenicline_validation/blob/docs/preprocessing-fiji/01_preprocessing/DISCREPANCIES.md).

## Segmentação e pós-processamento (`02_segmentation/`)

O pós-processamento é o que mais se afasta da tese: o filtro de volume, o closing e o passo em Z são todos diferentes. No treino, a anisotropia e o número de épocas não batem com a Tabela 3.3.

| Script | O que faz agora | O que a tese diz | Alteração | Secção |
| --- | --- | --- | --- | --- |
| `postprocessing/postprocessingfill.py` | Mantém objetos com 3 000–70 000 voxels (`MIN_VOL = 3000`) | 5 000–70 000 voxels | `MIN_VOL = 5000` | 3.7.1 |
| `postprocessing/postprocessingfill.py` | Closing 2D com uma cruz 3×3, uma iteração (≈ raio 1) | Closing 2D slice a slice com disco de raio 3 px | Usar `disk(3)` como elemento estruturante | 3.7.1 |
| `postprocessing/postprocessingfill.py` | Em Z, faz OR de cada slice com os vizinhos (dilatação de 1 px) e guarda o resultado dilatado | Dilatação de 1 px só em Z, depois interseção com a máscara original | Aplicar a interseção. Ver a contradição 3 abaixo: tal como está escrita, a interseção anula a dilatação | 3.7.1 |
| `postprocessing/postprocessingfill.py` | Mantém os IDs originais das labels | Reetiquetar de 1 a N | Acrescentar `relabel_sequential` antes de guardar | 3.7.1 |
| `train_stardist3D3dez.py` | Anisotropia (2.7, 1, 1) | Anisotropia (2.5, 1, 1) | `ANISO = (2.5, 1, 1)`, ou corrigir a Tabela 3.3 para 2.7 se foi esse o modelo avaliado | 3.6.1, Tab. 3.3 |
| `train_stardist3D3dez.py` | `train_epochs = 400` | 22 épocas de 100 passos | Confirmar se o treino parou às 22 épocas; fixar o valor que produziu o modelo avaliado | Tab. 3.3 |
| `dataaumentation/dataumentations3Dlatest.py` | `ANISO_FACTOR = 2.73` | Voxel 0,1 × 0,1 × 0,25 µm, ou seja 2,5 | Usar 2,5, ou explicar na tese de onde vem 2,73 | 3.4.3, Tab. 3.2 |
| `trainswincell2dezcomval.py` | `max_epochs = 300` | 70 épocas (parado cedo) | Documentar no script que o treino foi parado às 70 épocas | Tab. 3.4 |

Conferem com a tese: a inferência SwinCell (janela 256 × 256 × 64, overlap 75 %, cellprob −5, min\_size 2 500), os parâmetros de augmentation 3D (±30°, 0,9–1,1, elástica 15 / 4, ruído 0,02, gamma 0,7–1,3, 30 volumes, 20 % validação) e o StarDist (96 raios, grelha (1, 2, 2), patch 32 × 128 × 128, lr 3 × 10⁻⁴).

## Classificação (`03_classification/`)

O classificador usa os percentis e o shell de uma versão anterior. Pela tese (4.3), a Tabela 4.3 vem de DsRed P75 com Otsu por bloco e in situ P85 com shell \[−5, +3\] px e Otsu global. Antes de mudar, confirmar que os CSV em `Results coexpression/` foram gerados com os valores da tese.

| Script | O que faz agora | O que a tese diz | Alteração | Secção |
| --- | --- | --- | --- | --- |
| `overlap_full.py` | DsRed: `P_DSRED_BRIGHT = 85` | DsRed P75, percentil do slice inteiro | `P_DSRED_BRIGHT = 75` | 4.3 (Fig. 4.11, 4.12) |
| `overlap_full.py` | In situ: `P_INSITU_BRIGHT = 90` | In situ P85 | `P_INSITU_BRIGHT = 85` | 4.3 (Fig. 4.15) |
| `overlap_full.py` | Shell só para fora do núcleo: 0 < d ≤ 5 px (`RING_PX = 5`) | Shell \[−5, +3\] px: 5 px para dentro da borda do núcleo e 3 px para fora, limitado pelo Voronoi | Mudar `voronoi_shell_2d` para aceitar um offset interior e um exterior; usar −5 e +3 | 3.8.2 (Fig. 3.11), 4.3 |
| `overlap_full.py` | Otsu do in situ calculado por bloco | Otsu global para o in situ | Juntar os `pct_bright_insitu` de todos os blocos e calcular um único limiar | 4.3 |
| `overlap_full.py` | Não exclui células com fração zero antes do Otsu | Parâmetros finais incluem "zero-exclusion" | Excluir `pct_bright == 0` do cálculo do limiar | 4.3 |
| `overlap_full.py` | CSV com label e métricas, sem coordenadas | CSV com ID, coordenadas, features e labels binárias | Acrescentar o centróide 3D de cada célula | 3.8.2 (passo 7) |
| `classificationdsred.py` | `MANUAL_BEST_PERCENTILE = 85`, `OPTIMIZATION_METHOD = "otsu_global"` | P75, slice inteiro, Otsu por bloco | Mudar para 75 e Otsu por bloco | 4.3 |
| `dsredpermutation.py` | Classificador principal P85 + Otsu | P75 + Otsu por bloco | Mudar para P75 e Otsu por bloco | 4.3 |

O Otsu por bloco do DsRed já está certo em `overlap_full.py`. `densitymap.py` usa σ = 30 px, como na Fig. 4.20.

## Funcional 2P (`05_functional_2p/`)

A parte funcional está quase toda de acordo com a tese. A diferença principal é o percentil do DsRed num dos dois scripts de colocalização, e o resto são versões antigas por arquivar.

| Script | O que faz agora | O que a tese diz | Alteração | Secção |
| --- | --- | --- | --- | --- |
| `colocalizationv2.py` | `DSRED_PERCENTILE = 85` | P80 nos ROIs funcionais | Mudar para 80 (`2photongaba.py` já usa 80) e deixar só um dos dois scripts | 3.9.2 |
| `registrationfinal2.py`, `registofinal3.py`, `regsitration0202.py` | Versões SimpleITK / anteriores do registo | Registo escolhido: ANTs SyN (afim + SyN, suavização 1,0 anatómico e 0,5 funcional) | Manter `registofinal4deform.py` (confere com a Tab. A.8) e arquivar os outros | 3.9.2, A.7.3 |
| `Motioncorrection suite2p manel updated.py` | Registo rígido, `do_bidiphase = False` | Tab. A.7: não-rígido, two-step, `do_bidiphase = True` | Arquivar; usar `motioncorrectionfull.py`, que confere com a Tab. A.7 | A.7.1 |

Conferem com a tese: `motioncorrectionfull.py` e `Run motion correction1812 updated.py` (Tab. A.7), o stitching 3D em `02_segmentation/3. 3Dstiching.ipynb` (área 25–300 px, aspeto ≤ 1,5, centróide ≤ 3 px, overlap ≥ 0,5, 2–8 planos) e `2photongaba.py` (P80). O notebook de stitching deve passar para `05_functional_2p/`.

## Scripts em falta

A tese descreve quatro passos que não têm código no repositório. Sem o primeiro e o terceiro, não dá para reproduzir a Tabela 4.3 a partir do repositório.

| Passo | O que a tese diz | O que existe | Secção |
| --- | --- | --- | --- |
| Segmentação do cérebro inteiro em blocos | Cellpose `cyto3` afinado, restauro `denoise`, diameter ≈ 45, cellprob −4, flow3D smooth 3, em blocos de z sobrepostos; labels juntas por correspondência na zona de sobreposição | Nada. Só o modelo em `trained_models/`. A classificação corre bloco a bloco e nenhum script remove células contadas em dois blocos | 3.7 |
| Augmentation 2D para o Cellpose | Rotação ±30°, escala 0,9–1,1, flips, elástica, perspetiva, pincushion, motion blur; 15 variantes por imagem | Só a versão 3D (`dataumentations3Dlatest.py`) | 3.6, Tab. 3.1 |
| Métricas de fidelidade | Sensibilidade, especificidade, precisão, F1, odds ratio e teste exato de Fisher por peixe | Nenhum script identificado; `densitymap.py` só faz mapas | 3.8.4, Tab. 4.3 |
| Histogram matching original | Correção em Z por histogram matching ao slice do meio | Só a reconstrução do PR #1 | 3.5 |

## Contradições dentro da tese

Estes pontos têm de ser decididos na tese antes de alterar o código, porque a tese diz duas coisas diferentes.

1. **Percentil do DsRed.** A 4.3 escolhe P75; a 3.9.2 fala do "P85 used in the confocal pipeline". Proposta: P75 em todo o lado e corrigir a 3.9.2.
2. **Âmbito do Otsu.** A 3.8.2 diz que o Otsu é aplicado uma vez a todas as células do cérebro; a 4.3 escolhe Otsu por bloco para o DsRed e global para o in situ. Proposta: corrigir a 3.8.2 para refletir a 4.3.
3. **Passo em Z do pós-processamento.** A 3.7.1 diz dilatação de 1 px em Z seguida de interseção com a máscara original. Isso devolve a máscara original, portanto o passo não faz nada. O código guarda a dilatação. Falta decidir qual é a operação real e reescrever a frase.
4. **Shell do in situ no fluxo GPU.** O passo 4 da 3.8.2 diz "extranuclear pixels within 5 pixels" (o que o código faz); a 4.3 escolhe \[−5, +3\] px. Proposta: atualizar o passo 4.
5. **Motion correction.** A 3.9.2 diz registo rígido dentro do plano; a Tab. A.7 diz `nonrigid = True`. Os scripts seguem a Tab. A.7. Proposta: corrigir o texto da 3.9.2.
6. **Imagem.** As notas do protocolo dizem voxel 0,065 µm e stitching no BigStitcher; a tese diz 0,05 µm (3.4.3) e stitching no ZEN (3.4.4). Confirmar com os metadados dos ficheiros `.czi`.
