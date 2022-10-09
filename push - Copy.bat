cd /d W:\Documents\ElectronDensity.jl

git config --global user.email "pxshen@alumni.stanford.edu"
git config --global user.name "Paul Shen"
set GIT_SSL_NO_VERIFY=true

xcopy /s /Y W:\Documents\ElectronDensity.jl\docs\build\ W:\Documents\ElectronDensity.jl\docs\
git add -A
git commit -m "some message"
git push https://paulxshen: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@github.com/paulxshen/ElectronDensity.jl
