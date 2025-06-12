#!/usr/bin/env python3
"""
MAIN - EXECUTOR DO ANALISADOR TME GÁSTRICO
==========================================

Script principal para executar a análise exploratória TME expandida.
Este script permite execução independente do sistema de treinamento,
com opções flexíveis de configuração e saída detalhada.

Uso:
    python main_analisador_tme.py --data_path data --sample_size 100
    python main_analisador_tme.py --help

Autor: Baseado em Lou et al. (2025), Kather et al. (2019), Mandal et al. (2025)
"""

import os
import sys
import argparse
import time
from pathlib import Path
import json
import warnings

# Suprimir warnings desnecessários
warnings.filterwarnings('ignore')

# Importar o analisador TME
try:
    from sistema_analisador_imagens_tme import run_exploratory_analysis_before_training, TMEGastricAnalyzer
    print("✅ Analisador TME importado com sucesso!")
except ImportError as e:
    print(f"❌ Erro ao importar analisador TME: {e}")
    print("📁 Certifique-se de que o arquivo 'analisador_imagens.py' está no mesmo diretório")
    sys.exit(1)


def validate_data_structure(data_path: str) -> bool:
    """
    Valida se a estrutura de dados está correta para análise TME.
    
    Estrutura esperada:
    data/
    ├── train/
    │   ├── ADI/
    │   ├── DEB/
    │   ├── LYM/
    │   ├── MUC/
    │   ├── MUS/
    │   ├── NOR/
    │   ├── STR/
    │   └── TUM/
    ├── val/ (opcional)
    └── test/ (opcional)
    """
    data_path = Path(data_path)
    
    print(f"🔍 Validando estrutura de dados em: {data_path}")
    
    # Verificar se diretório existe
    if not data_path.exists():
        print(f"❌ Diretório não encontrado: {data_path}")
        return False
    
    # Verificar se existe pelo menos train/
    train_path = data_path / "train"
    if not train_path.exists():
        print(f"❌ Diretório 'train' não encontrado: {train_path}")
        print("💡 Estrutura esperada: data/train/[CLASSE]/imagens.png")
        return False
    
    # Classes TME esperadas
    expected_classes = ['ADI', 'DEB', 'LYM', 'MUC', 'MUS', 'NOR', 'STR', 'TUM']
    found_classes = []
    missing_classes = []
    
    print(f"📂 Verificando classes TME em: {train_path}")
    
    for class_name in expected_classes:
        class_path = train_path / class_name
        if class_path.exists():
            # Contar imagens
            images = list(class_path.glob("*.png")) + list(class_path.glob("*.jpg")) + list(class_path.glob("*.jpeg"))
            if images:
                found_classes.append(class_name)
                print(f"   ✅ {class_name}: {len(images)} imagens encontradas")
            else:
                missing_classes.append(class_name)
                print(f"   ⚠️  {class_name}: Diretório existe mas sem imagens")
        else:
            missing_classes.append(class_name)
            print(f"   ❌ {class_name}: Diretório não encontrado")
    
    print(f"\n📊 Resumo da validação:")
    print(f"   Classes encontradas: {len(found_classes)}/8")
    print(f"   Classes disponíveis: {found_classes}")
    
    if missing_classes:
        print(f"   Classes ausentes: {missing_classes}")
    
    # Validação mínima: pelo menos 4 classes
    if len(found_classes) < 4:
        print(f"\n❌ Estrutura de dados insuficiente!")
        print(f"   Mínimo necessário: 4 classes TME")
        print(f"   Encontradas: {len(found_classes)} classes")
        return False
    
    print(f"\n✅ Estrutura de dados validada com sucesso!")
    return True


def print_analysis_banner():
    """Imprime banner inicial da análise"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                     ANALISADOR TME GÁSTRICO - VERSÃO EXPANDIDA              ║
║                                                                              ║
║  🔬 Análise Exploratória Quantitativa do Microambiente Tumoral              ║
║  📚 Baseado em Literatura Científica Recente                                ║
║  🎯 Fundamentação para Otimização de Modelos de IA                          ║
║                                                                              ║
║  Referências Principais:                                                     ║
║  • Lou et al. (2025): HMU-GC-HE-30K dataset challenges                      ║
║  • Kather et al. (2019): TME classification difficulties                    ║
║  • Mandal et al. (2025): Nuclear morphology variability                     ║
║  • Vahadane et al. (2016): H&E stain separation methods                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def parse_arguments():
    """Parse argumentos da linha de comando"""
    parser = argparse.ArgumentParser(
        description='Analisador TME Gástrico - Análise Exploratória Expandida',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python main_analisador_tme.py --data_path data --sample_size 100
  python main_analisador_tme.py --data_path /caminho/para/dados --sample_size 50 --quick
  python main_analisador_tme.py --data_path data --sample_size 200 --verbose --save_plots
        """
    )
    
    # Argumentos principais
    parser.add_argument('--data_path', type=str, default='data',
                       help='Caminho para dados organizados (default: data)')
    
    parser.add_argument('--sample_size', type=int, default=100,
                       help='Número de imagens por classe para análise (default: 100)')
    
    # Opções de execução
    parser.add_argument('--quick', action='store_true',
                       help='Execução rápida com sample_size reduzido (50 imagens)')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Saída detalhada com informações adicionais')
    
    parser.add_argument('--save_plots', action='store_true',
                       help='Salvar todos os gráficos de análise')
    
    # Opções de validação
    parser.add_argument('--validate_only', action='store_true',
                       help='Apenas validar estrutura de dados (não executar análise)')
    
    parser.add_argument('--skip_validation', action='store_true',
                       help='Pular validação da estrutura de dados')
    
    # Opções de saída
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Diretório para salvar resultados (default: diretório atual)')
    
    parser.add_argument('--no_save', action='store_true',
                       help='Não salvar resultados em arquivo')
    
    return parser.parse_args()


def setup_environment(args):
    """Configura ambiente de execução"""
    
    # Ajustar sample_size para execução rápida
    if args.quick:
        args.sample_size = min(args.sample_size, 50)
        print(f"🚀 Modo rápido ativado: sample_size = {args.sample_size}")
    
    # Criar diretório de saída se necessário
    output_path = Path(args.output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Diretório de saída criado: {output_path}")
    
    # Configurar matplotlib para salvar plots se solicitado
    if args.save_plots:
        import matplotlib
        matplotlib.use('Agg')  # Backend não-interativo
        print(f"📊 Plots serão salvos no diretório: {output_path}")
    
    return args


def print_execution_summary(args):
    """Imprime resumo da configuração de execução"""
    print(f"\n⚙️  CONFIGURAÇÃO DE EXECUÇÃO:")
    print(f"{'─' * 50}")
    print(f"📂 Caminho dos dados: {args.data_path}")
    print(f"🎯 Sample size: {args.sample_size} imagens por classe")
    print(f"📊 Salvar plots: {'✅' if args.save_plots else '❌'}")
    print(f"📄 Salvar resultados: {'❌' if args.no_save else '✅'}")
    print(f"📁 Diretório de saída: {args.output_dir}")
    print(f"💬 Modo verbose: {'✅' if args.verbose else '❌'}")
    print(f"🚀 Modo rápido: {'✅' if args.quick else '❌'}")
    print(f"{'─' * 50}")


def print_results_summary(results: dict, execution_time: float):
    """Imprime resumo dos resultados da análise"""
    
    print(f"\n🎯 RESUMO DOS RESULTADOS DA ANÁLISE")
    print(f"{'═' * 70}")
    
    # Informações gerais
    exec_summary = results.get('execution_summary', {})
    print(f"⏱️  Tempo de execução: {execution_time:.1f} segundos")
    print(f"📂 Dataset analisado: {exec_summary.get('dataset_path', 'N/A')}")
    print(f"📊 Sample size utilizado: {exec_summary.get('sample_size', 'N/A')}")
    
    # Validação da literatura
    analysis_results = results.get('analysis_results', {})
    lit_validation = analysis_results.get('literature_validation', {})
    
    validated_problems = []
    total_problems = len(lit_validation)
    
    print(f"\n📚 VALIDAÇÃO DA LITERATURA:")
    print(f"{'─' * 40}")
    
    for problem, validation in lit_validation.items():
        problem_name = problem.replace('_', ' ').title()
        if validation.get('validated', False):
            validated_problems.append(problem)
            confidence = validation.get('confidence', 0)
            source = validation.get('literature_source', 'N/A')
            print(f"✅ {problem_name}: Validado (confiança: {confidence:.3f})")
            print(f"   📖 Fonte: {source}")
        else:
            print(f"❌ {problem_name}: Não validado")
    
    print(f"\n📈 Score de validação: {len(validated_problems)}/{total_problems} problemas confirmados")
    
    # Dificuldade diagnóstica
    diagnostic_difficulty = analysis_results.get('diagnostic_difficulty', {})
    difficulty_scores = diagnostic_difficulty.get('diagnostic_difficulty_score', {})
    
    if difficulty_scores:
        print(f"\n🎯 RANKING DE DIFICULDADE DIAGNÓSTICA:")
        print(f"{'─' * 50}")
        
        ranked_difficulty = sorted(difficulty_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (class_name, score) in enumerate(ranked_difficulty[:5]):  # Top 5
            priority = "🔴 ALTA" if score > 0.7 else "🟡 MÉDIA" if score > 0.5 else "🟢 BAIXA"
            print(f"{i+1}. {class_name}: {score:.3f} - Prioridade {priority}")
    
    # Recomendações
    recommendations = results.get('recommendations', {})
    immediate_actions = recommendations.get('immediate_actions', [])
    medium_term = recommendations.get('medium_term_optimizations', [])
    
    print(f"\n💡 RECOMENDAÇÕES DE OTIMIZAÇÃO:")
    print(f"{'─' * 50}")
    print(f"🚨 Ações imediatas: {len(immediate_actions)}")
    print(f"⚡ Otimizações médio prazo: {len(medium_term)}")
    
    for i, action in enumerate(immediate_actions[:3], 1):  # Top 3
        print(f"  {i}. {action['action']}")
        print(f"     📈 Melhoria esperada: {action['expected_improvement']}")
    
    print(f"\n✅ Análise concluída com sucesso!")
    print(f"{'═' * 70}")


def save_results_to_files(results: dict, args, execution_time: float):
    """Salva resultados em arquivos"""
    
    if args.no_save:
        print(f"📄 Salvamento de resultados desabilitado")
        return
    
    output_path = Path(args.output_dir)
    
    # 1. Salvar resultados JSON completos
    json_file = output_path / 'tme_analysis_results.json'
    
    # Adicionar metadados de execução
    results['execution_metadata'] = {
        'execution_time_seconds': execution_time,
        'arguments': vars(args),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'version': '2.0_expanded'
    }
    
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"💾 Resultados salvos em: {json_file}")
    except Exception as e:
        print(f"❌ Erro ao salvar JSON: {e}")
    
    # 2. Salvar relatório resumido em texto
    txt_file = output_path / 'tme_analysis_summary.txt'
    
    try:
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO DE ANÁLISE TME GÁSTRICO\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Data/Hora: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {args.data_path}\n")
            f.write(f"Sample size: {args.sample_size}\n")
            f.write(f"Tempo de execução: {execution_time:.1f}s\n\n")
            
            # Validações da literatura
            lit_validation = results.get('analysis_results', {}).get('literature_validation', {})
            validated_count = sum(1 for v in lit_validation.values() if v.get('validated', False))
            
            f.write(f"VALIDAÇÃO DA LITERATURA:\n")
            f.write(f"Problemas validados: {validated_count}/{len(lit_validation)}\n\n")
            
            for problem, validation in lit_validation.items():
                status = "✓" if validation.get('validated', False) else "✗"
                confidence = validation.get('confidence', 0)
                f.write(f"{status} {problem.replace('_', ' ').title()}: {confidence:.3f}\n")
            
            # Recomendações principais
            f.write(f"\nRECOMENDAÇÕES PRINCIPAIS:\n")
            immediate_actions = results.get('recommendations', {}).get('immediate_actions', [])
            for i, action in enumerate(immediate_actions, 1):
                f.write(f"{i}. {action['action']}\n")
                f.write(f"   Melhoria esperada: {action['expected_improvement']}\n\n")
        
        print(f"📝 Relatório resumido salvo em: {txt_file}")
        
    except Exception as e:
        print(f"❌ Erro ao salvar relatório: {e}")
    
    # 3. Salvar CSV com métricas por classe (se disponível)
    try:
        analysis_results = results.get('analysis_results', {})
        
        if 'diagnostic_difficulty' in analysis_results:
            import pandas as pd
            
            difficulty_scores = analysis_results['diagnostic_difficulty'].get('diagnostic_difficulty_score', {})
            robustness_scores = analysis_results.get('morphological_robustness', {}).get('overall_robustness_score', {})
            
            # Criar DataFrame com métricas por classe
            df_data = []
            for class_name in ['ADI', 'DEB', 'LYM', 'MUC', 'MUS', 'NOR', 'STR', 'TUM']:
                row = {
                    'Classe': class_name,
                    'Dificuldade_Diagnostica': difficulty_scores.get(class_name, 0),
                    'Robustez_Morfologica': robustness_scores.get(class_name, 0),
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            csv_file = output_path / 'tme_class_metrics.csv'
            df.to_csv(csv_file, index=False)
            print(f"📊 Métricas por classe salvas em: {csv_file}")
            
    except Exception as e:
        print(f"⚠️  Aviso: Não foi possível salvar CSV de métricas: {e}")


def main():
    """Função principal"""
    
    # Parse argumentos
    args = parse_arguments()
    
    # Imprimir banner
    print_analysis_banner()
    
    # Setup do ambiente
    args = setup_environment(args)
    
    # Imprimir configuração
    print_execution_summary(args)
    
    # Validar estrutura de dados
    if not args.skip_validation:
        print(f"\n🔍 VALIDANDO ESTRUTURA DE DADOS...")
        if not validate_data_structure(args.data_path):
            print(f"\n❌ Validação falhou. Corrija a estrutura de dados e tente novamente.")
            print(f"\n💡 Estrutura esperada:")
            print(f"   data/train/ADI/*.png")
            print(f"   data/train/DEB/*.png")
            print(f"   data/train/LYM/*.png")
            print(f"   ...")
            sys.exit(1)
    
    # Se apenas validação, sair aqui
    if args.validate_only:
        print(f"\n✅ Validação concluída com sucesso!")
        sys.exit(0)
    
    # Executar análise principal
    print(f"\n🚀 INICIANDO ANÁLISE EXPLORATÓRIA...")
    print(f"{'═' * 80}")
    
    start_time = time.time()
    
    try:
        # Executar análise completa
        results = run_exploratory_analysis_before_training(
            config_or_data_path=args.data_path,
            sample_size=args.sample_size
        )
        
        execution_time = time.time() - start_time
        
        # Imprimir resumo dos resultados
        print_results_summary(results, execution_time)
        
        # Salvar resultados
        save_results_to_files(results, args, execution_time)
        
        # Mensagem final
        print(f"\n🎉 ANÁLISE CONCLUÍDA COM SUCESSO!")
        print(f"⏱️  Tempo total: {execution_time:.1f} segundos")
        
        if not args.no_save:
            print(f"📁 Resultados salvos em: {args.output_dir}")
            print(f"📄 Arquivo principal: tme_analysis_results.json")
            print(f"📝 Relatório resumido: tme_analysis_summary.txt")
        
        print(f"\n💡 Para implementar as otimizações sugeridas:")
        print(f"   1. Revise o arquivo tme_analysis_results.json")
        print(f"   2. Implemente as 'immediate_actions' primeiro")
        print(f"   3. Use o 'validation_framework' para medir melhorias")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Análise interrompida pelo usuário")
        sys.exit(1)
        
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"\n❌ ERRO DURANTE A ANÁLISE:")
        print(f"   Erro: {str(e)}")
        print(f"   Tempo até erro: {execution_time:.1f} segundos")
        
        if args.verbose:
            import traceback
            print(f"\n🔍 DETALHES DO ERRO (verbose):")
            traceback.print_exc()
        
        print(f"\n💡 DICAS PARA SOLUCIONAR:")
        print(f"   1. Verifique se todas as dependências estão instaladas")
        print(f"   2. Confirme que as imagens estão no formato correto (PNG/JPG)")
        print(f"   3. Tente com --sample_size menor (ex: 20)")
        print(f"   4. Use --verbose para mais detalhes do erro")
        
        sys.exit(1)


if __name__ == "__main__":
    main()