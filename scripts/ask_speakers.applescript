-- Ask for the number of voices before processing one audio file.
-- Returns "auto", a number from 1 to 20, or "__CANCEL__".
on run argv
    set recordingName to item 1 of argv
    set previousAnswer to "2"
    repeat
        try
            set reply to display dialog ("Запись: " & recordingName & return & return & "Сколько человек говорят в записи?" & return & "Йоу 🌺 Введи число от 1 до 20. Если количество спикеров неизвестно, введи auto.") with title "Whisper — количество спикеров" default answer previousAnswer buttons {"Отмена", "Начать"} default button "Начать" cancel button "Отмена"
            set answerText to text returned of reply
            set previousAnswer to answerText
            if answerText is "auto" or answerText is "авто" or answerText is "" then return "auto"
            try
                set numberOfSpeakers to answerText as integer
                if (numberOfSpeakers >= 1) and (numberOfSpeakers <= 20) and ((numberOfSpeakers as text) is answerText) then
                    return (numberOfSpeakers as text)
                end if
            end try
            display alert "Некорректное количество" message "Введите целое число от 1 до 20 или слово auto." as warning
        on error number -128
            return "__CANCEL__"
        end try
    end repeat
end run
