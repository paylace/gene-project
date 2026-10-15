param(
    [ValidateSet('status', 'fetch', 'pull', 'push')]
    [string]$Action = 'status'
)

$ErrorActionPreference = 'Stop'
$gitWorkspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$gitCommand = Get-Command git -CommandType Application -ErrorAction SilentlyContinue
if ($gitCommand) {
    $gitSyncExecutable = $gitCommand.Source
} else {
    $gitSyncExecutable = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe'
}
if (-not (Test-Path -LiteralPath $gitSyncExecutable)) {
    throw 'Git을 찾지 못했습니다. Git for Windows 설치 후 다시 실행하세요.'
}

function Invoke-ProjectGit {
    param([string[]]$GitArguments)
    & $gitSyncExecutable -C $gitWorkspace @GitArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Git 명령 실패(종료 코드 $LASTEXITCODE). 원자료와 로컬 변경은 그대로 보존했습니다."
    }
}

$remoteUrl = Invoke-ProjectGit -GitArguments @('remote', 'get-url', 'origin')
if ($remoteUrl.Trim() -ne 'https://github.com/paylace/gene-project.git') {
    throw 'origin이 paylace/gene-project와 다릅니다. 원격 설정을 먼저 확인하세요.'
}

switch ($Action) {
    'status' {
        Invoke-ProjectGit -GitArguments @('status', '--short', '--branch')
        Invoke-ProjectGit -GitArguments @('branch', '-vv')
        Invoke-ProjectGit -GitArguments @('remote', '-v')
    }
    'fetch' {
        Invoke-ProjectGit -GitArguments @('fetch', '--prune', 'origin')
        Invoke-ProjectGit -GitArguments @('status', '--short', '--branch')
        Write-Output '원격 변경 확인 완료. 작업 파일에는 병합하지 않았습니다.'
    }
    'pull' {
        $trackedChanges = Invoke-ProjectGit -GitArguments @('status', '--porcelain=v1', '--untracked-files=no')
        if ($trackedChanges) {
            throw '커밋하지 않은 변경이 있습니다. 내용을 검토하고 커밋한 뒤 pull을 실행하세요. 자동 stash/덮어쓰기는 하지 않습니다.'
        }
        Invoke-ProjectGit -GitArguments @('pull', '--ff-only')
    }
    'push' {
        $currentBranch = Invoke-ProjectGit -GitArguments @('branch', '--show-current')
        if (-not $currentBranch -or $currentBranch.Trim() -in @('main', 'master')) {
            throw '작업 브랜치에서만 업로드합니다. main/master 또는 detached HEAD의 push는 거부합니다.'
        }
        Invoke-ProjectGit -GitArguments @('push', '--set-upstream', 'origin', 'HEAD')
        Write-Output '커밋된 변경의 업로드 완료. 커밋하지 않은 변경은 업로드하지 않았습니다.'
    }
}
